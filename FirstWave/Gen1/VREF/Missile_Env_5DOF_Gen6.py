#!/usr/bin/env python
#-*- coding: utf-8 -*-
########################################################################            
########                                        _..._           ########      
########                    _______          .-'_..._''.        ########      
########                    \  ___ `'.     .' .'      '.\       ########      
########                _.._ ' |--.\  \   / .'                  ########      
########              .' .._|| |    \  ' . '                    ########      
########              | '    | |     |  '| |                    ########      
########            __| |__  | |     |  || |                    ########      
########           |__   __| | |     ' .'. '                    ########      
########              | |    | |___.' /'  \ '.          .       ########      
########              | |   /_______.'/    '. `._____.-'/       ########      
########              | |   \_______|/       `-.______ /        ########      
########              | |                             `         ########      
########              |_|                                       ########      
########           .---.                                        ########      
########           |   |          /|                            ########      
########           |   |          ||                            ########      
########           |   |          ||                            ########      
########           |   |    __    ||  __                        ########      
########           |   | .:--.'.  ||/'__ '.                     ########      
########           |   |/ |   \ | |:/`  '. '                    ########      
########           |   |`" __ | | ||     | |                    ########      
########           |   | .'.''| | ||\    / '                    ########      
########           '---'/ /   | |_|/\'..' /                     ########      
########                \ \._,\ '/'  `'-'`                      ########      
########                 `--'  `"                               ########      
########################################################################
#
#   ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗             
#   ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗            
#   ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝            
#   ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗            
#    ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║            
#     ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝            
#                                                                  
#    ██████╗ ██╗   ██╗██╗██████╗  █████╗ ███╗   ██╗ ██████╗███████╗
#   ██╔════╝ ██║   ██║██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝
#   ██║  ███╗██║   ██║██║██║  ██║███████║██╔██╗ ██║██║     █████╗  
#   ██║   ██║██║   ██║██║██║  ██║██╔══██║██║╚██╗██║██║     ██╔══╝  
#   ╚██████╔╝╚██████╔╝██║██████╔╝██║  ██║██║ ╚████║╚██████╗███████╗
#    ╚═════╝  ╚═════╝ ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
#                                                                  
# Pseudo 5 DOF 3dimensional Missile_Environment for Reinforcement Learning
#   Version --5DOF
#   Created by Hong Daseon
#   Input has command Vvector derivative

import PSpincalc as spin
import DaseonTypesNtf as Daseon
from pyquaternion import Quaternion
from DaseonTypesNtf import Vector3, DCM5DOF
import math as m
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import time
import copy
import pdb

#Missile coordinate trans Model
Debug = False
gAcc = 0. #9.806


class Missile_3D:
    
    def __init__(self, scavel, initPos, initAtt, dt):
        self.initval        = [scavel, initPos.x, initPos.y, initPos.z, initAtt.z, initAtt.y, dt]
        self.scavel         = scavel
        self.bodyVelDir     = Vector3(self.scavel,0.,0.)
        self.pos            = initPos
        self.datt           = Vector3(0.,0.,0.)
        self.att            = initAtt
        self.Cnb            = DCM5DOF(self.att)

        self.ControllerAccZ = Daseon.SecondOrder(5, 0.75, dt)
        self.ControllerAccY = Daseon.SecondOrder(5, 0.75, dt)
        self.acc            = Vector3(0.,0.,0.)
        self.dpos           = Cnb.rotate(self.bodyVelDir, 'inv')
        self.actuation      = Vector3(0.,0.,0.)
        
        self.dt             = dt

        self.reset_flag     = True

        self.IntegAtt_y     = Daseon.Integrator(self.att.y, dt)
        self.IntegAtt_z     = Daseon.Integrator(self.att.z, dt)

        self.IntegPos_x     = Daseon.Integrator(self.pos.x, dt)
        self.IntegPos_y     = Daseon.Integrator(self.pos.y, dt)
        self.IntegPos_z     = Daseon.Integrator(self.pos.z, dt)

    def simulate(self, acc):
        self.actuation.z    = self.ControllerAccZ.step(acc.z)
        self.actuation.y    = self.ControllerAccY.step(acc.y)
        self.datt.z         = self.actuation.y\
                                /self.scavel/m.cos(self.att.y)
        self.datt.y         = -(self.actuation.z+gAcc*m.cos(self.att.y))\
                                /self.scavel
        self.att.y          = self.IntegAtt_y.step(self.datt.y)
        self.att.z          = self.IntegAtt_z.step(self.datt.z)
        
        self.Cnb.update(self.att)

        self.dpos           = Cnb.rotate(self.bodyVelDir, 'inv')

        self.pos.x          = self.IntegPos_x.step(self.dpos.x)
        self.pos.y          = self.IntegPos_y.step(self.dpos.y)
        self.pos.z          = self.IntegPos_z.step(self.dpos.z)

        return self.dpos, self.pos

    def reset(self, _pos, _att, Vm, reset_flag):
        self.scavel         = Vm
        self.bodyVelDir     = Vector3(self.scavel,0.,0.)
        self.pos            = _pos
        self.att            = _att
        self.acc            = Vector3(0.,0.,0.)
        self.dt             = self.dt      

        self.Cnb.reset(_att)
        self.ControllerAzim.reset()
        self.ControllerElev.reset()
        self.IntegAtt_y.reset()
        self.IntegAtt_z.reset()
        self.IntegPos_x.reset()
        self.IntegPos_y.reset()
        self.IntegPos_z.reset()

        self.dpos           = self.Cnb.rotate(Vector3.cast([self.scavel, 0, 0]))
        
        #print('just after reset : ',self.Qnb)
        self.reset_flag     = reset_flag
    
    def __str__(self):
        nowpos = 'x : '+ format(self.pos.x,".2f")+ ' y : '+ format(self.pos.y,".2f")+ ' z : '+ format(self.pos.z,".2f")

        return nowpos

class Seeker:
    #prevR = 0
    def __init__(self, Missile, Target):
        self.Rvec       = Target.pos - Missile.pos
        self.Vvec       = Target.dpos - Missile.dpos

        self.direcVec   = Missile.dpos

        self.Target     = Target
        self.Missile    = Missile
        
        self.impactR    = 9999999

        LOSz, LOSy      = self.azimNelev(self.direcVec)
        self.LOS        = Vector3(0.,LOSy, LOSz)
        self.dLOS       = Vector3(0., 0., 0.) # body frame
        self.prevLOS    = Vector3(0.,LOSy, LOSz)

        Lookz, Looky    = self.azimNelev(Missile.Cnb.rotate(self.Rvec))
        self.Look       = Vector3(0., Looky, Lookz)
        self.pLook      = copy.deepcopy(self.Look)
        self.ppLook     = copy.deepcopy(self.pLook)

        self.firstrun   = True

        self.prev_Rm  = Vector3(9999999, 9999999, 9999999)

        self.t2go       = 600

    def angle(self, vec1, vec2):
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]      # dot product
        det = vec1[0]*vec2[1] - vec2[0]*vec1[1]      # determinant
        return m.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    def azimNelev(self, vec):
        azim = m.atan2( vec.y, vec.x)
        elev = m.atan2( -vec.z, m.sqrt( vec.x**2 + vec.y**2))
        return azim, elev

    def seek(self, t):
        def normL(LOSval):
            return (LOSval)/3.14

        def normLd(LOSdotval):
            return (LOSdotval)*10
        
        def normVm(Vval):
            return Vval/600

        def normLk(Vval):
            return Vval/1.57
        #pdb.set_trace()
        self.t2go       = 0
        #print('in seek : ',self.Missile.Qnb)
        self.Rvec       = self.Target.pos - self.Missile.pos
        self.Vvec       = self.Target.dpos - self.Missile.dpos
        self.direcVec   = self.Missile.dpos
                
        LOSz, LOSy      = self.azimNelev(self.direcVec)
        self.LOS        = Vector3(0.,LOSy, LOSz)
        if t == 0 : 
            self.prevLOS = copy.deepcopy(self.LOS)
            #print('t=0 detected')
        self.ppLook     = copy.deepcopy(self.pLook)
        self.pLook      = copy.deepcopy(self.Look) 
        Lookz, Looky    = self.azimNelev(Missile.Cnb.rotate(self.Rvec))
        self.Look       = Vector3(0., Looky, Lookz)

        RjxVj = np.cross(self.Rvec.vec, self.Vvec.vec)
        RjdRj = np.dot(self.Rvec.vec, self.Rvec.vec)
        Ldotn = RjxVj/RjdRj
        
        Ldotb = self.Missile.Qnb.inverse.rotate(Ldotn)
        self.dLOS = Vector3.cast(Ldotb)
        self.Missile.reset_flag = False
        Vvecb = Vector3.cast(self.Missile.Qnb.inverse.rotate(self.Vvec.vec))
        return self.Rvec.mag, self.Look, self.dLOS, self.Missile.scavel,\
                                                    np.array([  normLk(self.ppLook.y), normLk(self.pLook.y), normLk(self.Look.y),\
                                                                normLk(self.ppLook.z), normLk(self.pLook.z), normLk(self.Look.z)])
                                                                                
    def newStepStarts(self, t):
        if t != 0:
            self.prevLOS = copy.deepcopy(self.LOS)
            #print('prevValEngaged')

    def spit_reward(self, acc):
        
        OOR         = (self.Look.x < -1.57)|(self.Look.x > 1.57)|(self.Look.y < -1.57)|(self.Look.y > 1.57)|(self.Look.z < -1.57)|(self.Look.z > 1.57)|(self.Rvec.mag>20000)  # Out of range
        if OOR:
            Rf_1 = self.prev_Rm
            Rf = self.Missile.pos
            
            R3 = Rf - Rf_1
            A = R3
            B = (self.Target.pos - Rf_1) - R3
            
            if self.Rvec.mag < 50:

                self.impactR = (Vector3.cast(np.cross(A.vec,B.vec)).mag) / A.mag 

            else:
                self.impactR = self.Rvec.mag

            rwdR = copy.deepcopy(self.impactR)
            
            if Debug : pdb.set_trace()
            print(rwdR)
        else:
            self.prev_Rm = copy.deepcopy(self.Missile.pos)
            rwdR = copy.deepcopy(self.Rvec.mag)

        


        hit         = (self.impactR <2)

        step_reward  = np.array([-(acc.y**2),-(acc.z**2)])
         #0.01*-Rdot - 3*abs(self.LOS) - 1.2*abs(self.LOS)*500*abs(Ldot) + (2/(self.R / 8000))**2.5 - (self.R<10000)*self.R/5000 #-1000*abs(Ldot)   -self.R/10000  # - self.R/100
        
        mc_reward    =  - (rwdR)
        
        #reward = (not OOR)*reward -OOR*25
        
        return step_reward, mc_reward, (OOR), hit

