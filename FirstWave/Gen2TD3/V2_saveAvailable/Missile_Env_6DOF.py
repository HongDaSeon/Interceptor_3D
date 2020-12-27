#!/usr/bin/env python
#-*- coding: utf-8 -*-
################################################################################################################################################################################ -ohh`  #####################                
#                                                                                                                                                                             -ohNMMMMd`                    #
#                                                                                                                                                                         -ohNMMMMMmMMMd`                   #
#                                                                                                                                                                     -ohNMMMMNho-  yMMMd`                  #
#                                                                                                                                                                 -ohNMMMMNho-       yMMMd`                 #
#                                                                                                                                                             -ohNMMMMNho-            yMMMd`                #
#                                                                                                                                                         -+hNMMMMMdo-                 yMMMd`               #
#                                                                                                                                                     -+hNMMMMMdo:`                     yMMMd`              #
#                                                                                                                                                 -+hNMMMMMho-                           sMMMd`             #
#                                                                                                                                                `dMMMMho-                                sMMMd`            #
#                                                                                                                                                 `dMMMo                                   sMMMd`           #
#                                                                                                                                                  `dMMMs                                   sMMMd`          #
#                                                                                                                                                   `dMMMs                                   sMMMm`         #
#                                                                                                                                                    `dMMMo         `:osyyo/.                 sMMMm`        #
#                                                                                                                                                     `dMMMo      `sNMMMMMMMMh-                sMMMm`       #
#    MMMMMMMMo mMMMMNNds-    -yNMMMMNd-      sMMs                -MMh                                    `ddh                                          `dMMMo    `mMMMMMMMMMMMM/                sMMMm`      #
#    MMM/----. mMMo--+mMMy  oMMN+.`./h:      sMMy      -+osso/.  -MMh-+so:    :osso/`  .++:`/+`-+osso/. `+MMN++/  -+sso/.  `++/`:+/++-   :++.           `dMMMo   oMMMMMMMMMMMMMm                 sMMMm.     #
#    MMMyssss` mMM/   `mMM/.MMM-             sMMy      oysosmMM/ -MMNhydMMy -mMNsodMN/ /MMNMNN.oysosmMM/.yMMMyyo`dMMyohMMo .MMNNNN+NMN. -MMd             `dMMMs  oMMMMMMMMMMMMMm              ./ymMMMMm`    #
#    MMMhyyyy` mMM/    dMMo-MMM`             sMMy      .+syhmMMs -MMh   NMM.hMM/  `MMM`/MMh    .+syhmMMs `MMN   sMMo   mMM-.MMm`   :MMh dMN.              `dMMMs `mMMMMMMMMMMMM:          `/ymMMMMMms/`     #
#    MMM.      mMM/  `oMMm` mMMy`    /.      sMMy     :MMd:-oMMs -MMh  `NMM`hMM/  `MMN`/MMy   :MMd:-oMMs `MMN   oMMs   mMM-.MMm     oMMdMM:                `dMMMs `sNMMMMMMMMh-       `/smMMMMMms/`         #
#    MMM.      mMMmdmMMNs`  `yMMNdhdNM:      sMMNmmmmd-MMNssmMMs -MMNyyNMN/ .mMNysmMN/ /MMy   -MMNssmMMs  mMMhss`hMMysdMMo .MMm      hMMMs                  `dMMMs   -+ssso/`     `/smMMMMMms/`             #
#    ///`      //////:.       `:+oo+:.       -//////// .+o+:.//- `//::++/`    -+oo+:`  .//-    .+o+:.//-   :+o+:  ./ooo/`  `///      +MMd                    `dMMMs           ./smMMMMMms/`                 #
#                                                                                                                                   .NMN.                     `dMMMs      ./ymMMMMMms/`                     #
#                                                                                                                                   `..`                       `hMMMs `/smMMMMMmy/`                         #
#                                                                                                                                                               `hMMMmMMMMMmy/.                             #
#                                             T  H  E    M  O  T  I  O  N    T  E  C  H  N  O  L  O  G  Y    I  N  N  O  V  A  T  I  O  N  S                     `hMMMMmy/.                                 #
#                                                                                                                                                                 `yy/.                                     #
#                                                                                                                                                                                                           #
######################################################################################## /+:` ###############################################################################################################

# Pseudo 6 DOF 3dimensional Missile_Environment for Reinforcement Learning
#   Version --6DOF
#   Created by Hong Daseon
#   rotate around X is activated

import PSpincalc as spin
from pyquaternion import Quaternion
from DaseonTypes import Vector3
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

def euler_to_quaternion(att):

    roll = att.x
    pitch = att.y
    yaw = att.z

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def RotationQuaternion(att, option="n_b"):
    _qx = Quaternion(axis=[1, 0, 0], angle=att.x)
    _qy = Quaternion(axis=[0, 1, 0], angle=att.y)
    _qz = Quaternion(axis=[0, 0, 1], angle=att.z)
    return _qz*_qy*_qx


class Missile_3D:
    
    def __init__(self, scavel, initPos, initAtt, dt):
        self.initval        = [scavel, initPos.x, initPos.y, initPos.z, initAtt.z, initAtt.y, dt]
        self.scavel         = scavel

        self.pos            = initPos
        self.att            = initAtt
        self.Qnb            = RotationQuaternion(self.att)

        self.acc            = Vector3(0.,0.,0.)
        self.dpos           = Vector3.cast(self.Qnb.rotate([self.scavel, 0, 0]))
        
        self.Rforward       = Vector3(0.,0.,0.)
        self.Vforward       = Vector3(0.,0.,0.)

        self.dt             = dt
      
        self.reset_flag     = True



    def simulate(self, _acc):
        self.acc            = _acc
        
        self.Qnb            = RotationQuaternion(self.att)

        self.accN           = Vector3.cast(self.Qnb.rotate(self.acc.vec))

        #posRatee            = self.Qnb.rotate([self.scavel, 0, 0])
        self.dpos.x         = self.dpos.x + self.accN.x*self.dt
        self.dpos.y         = self.dpos.y + self.accN.y*self.dt
        self.dpos.z         = self.dpos.z + self.accN.z*self.dt

        self.pos.x          = self.pos.x + self.dpos.x*self.dt
        self.pos.y          = self.pos.y + self.dpos.y*self.dt
        self.pos.z          = self.pos.z + self.dpos.z*self.dt

        self.Rforward       = self.pos
        self.Vforward       = self.dpos

        RjxVj               = np.cross(self.Rforward.vec, self.Vforward.vec)
        RjdRj               = np.dot(self.Rforward.vec, self.Rforward.vec)
        Ldotn               = RjxVj/RjdRj
        phi                 = self.att.x
        the                 = self.att.y
        Jinv                = np.array([ [1,    m.sin(phi)*m.tan(the),     m.cos(phi)*m.tan(the)],\
                                         [0,    m.cos(phi),                -m.sin(phi)],\
                                         [0,    m.sin(phi)/m.cos(the),     m.cos(phi)/m.cos(the)] ])
        Attidot             = np.matmul(Jinv, Ldotn)

        self.att.x          = self.att.x + Attidot[0]*self.dt
        self.att.y          = self.att.y + Attidot[1]*self.dt
        self.att.z          = self.att.z + Attidot[2]*self.dt

        return self.dpos, self.pos

    def reset(self, _pos, _att, Vm, reset_flag):
        self.scavel         = Vm

        self.pos            = _pos

        self.att            = _att

        self.acc            = Vector3(0.,0.,0.)

        self.dt             = self.dt
        
        self.Qnb            = RotationQuaternion(self.att)

        self.dpos           = Vector3.cast(self.Qnb.rotate([self.scavel, 0, 0]))
        
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

        Lookz, Looky    = self.azimNelev(self.Rvec - self.direcVec)
        self.Look       = Vector3(0., Looky, Lookz)
        self.Aiming     = 0

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
            return (LOSdotval)*20
        
        def normVm(Vval):
            return Vval/abs(Vval) * (abs(Vval)-200)/400
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

        #Lookz, Looky    = self.azimNelev(Vector3.cast(self.Missile.Qnb.inverse.rotate(self.Rvec.vec)))
        #self.Look       = Vector3(0., Looky, Lookz)
        self.Look       = Vector3.cast(np.arctan2(np.cross(self.direcVec.vec,self.Rvec.vec),\
                                                    np.dot(self.direcVec.vec,self.Rvec.vec)))
        
        self.Aiming     = np.dot(self.Rvec.vec, self.direcVec.vec)/self.Rvec.mag/self.direcVec.mag
        RjxVj = np.cross(self.Rvec.vec, self.Vvec.vec)
        RjdRj = np.dot(self.Rvec.vec, self.Rvec.vec)
        Ldotn = RjxVj/RjdRj
        
        Ldotb = self.Missile.Qnb.inverse.rotate(Ldotn)
        self.dLOS = Vector3.cast(Ldotb)
        self.Missile.reset_flag = False
        return self.Rvec.mag, self.Look, self.dLOS, self.Missile.scavel,\
                                                                    np.array([  normVm(self.Vvec.x),\
                                                                                normVm(self.Vvec.y),\
                                                                                normVm(self.Vvec.z),\
                                                                                normLd(self.dLOS.x),\
                                                                                normLd(self.dLOS.y),\
                                                                                normLd(self.dLOS.z)])
                                                                                
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

