import skfuzzy as fuzz
import numpy as np
#import matplotlib as plt
from skfuzzy import control as ctrl
ang=ctrl.Antecedent(np.arange(-44,46,1), 'Angulo')
vel=ctrl.Antecedent(np.arange(-1.5,2,0.5), 'Velocidad Angular')
velpt=ctrl.Consequent(np.arange(-3,4,1), 'Velocidad')
#Funciones de pertenencia para angulo
ang['ng']=fuzz.trapmf(ang.universe, [-50,-45, -30, -15])
ang['np']=fuzz.trimf(ang.universe, [-30,  -15, 0])
ang['z']=fuzz.trimf(ang.universe, [-15, 0, 15])
ang['pp']=fuzz.trimf(ang.universe, [0 , 15, 30])
ang['pg']=fuzz.trapmf(ang.universe, [15, 30, 45, 50])

#Funciones de pertenencia para velocidad angular
vel['ng']=fuzz.trapmf(vel.universe, [-2,-1.5,  -1, -0.5])
vel['np']=fuzz.trimf(vel.universe, [-1, -0.5, 0])
vel['z']=fuzz.trimf(vel.universe, [-0.5, 0, 0.5])
vel['pp']=fuzz.trimf(vel.universe, [0, 0.5, 1])
vel['pg']=fuzz.trapmf(vel.universe, [0.5, 1,1.5,2])

#Funciones de pertenencia para velocidad de la plataforma
velpt['ng']=fuzz.trapmf(velpt.universe, [-4 ,-3, -2, -1])
velpt['np']=fuzz.trimf(velpt.universe, [-2, -1, 0])
velpt['z']=fuzz.trimf(velpt.universe, [-1, 0, 1])
velpt['pp']=fuzz.trimf(velpt.universe, [0, 1, 2])
velpt['pg']=fuzz.trapmf(velpt.universe, [1, 2,3,4])
#Mostar graficas de las funciones de pertenencia
ang.view()
vel.view()
velpt.view()
#Reglas difusas
rule1=ctrl.Rule((ang['ng'] & vel['z']) | (ang['z'] & vel['ng']),velpt['ng'])
rule2=ctrl.Rule((ang['np'] & vel['z']) | (ang['z'] & vel['np']),velpt['np'])
rule3=ctrl.Rule(((ang['np'] & vel['pp']) | (ang['z'] & vel['z']) | (ang['pp'] & vel['np'])),velpt['z'])
rule4=ctrl.Rule((ang['pp'] & vel['z']) | (ang['z'] & vel['pp']),velpt['pp'])
rule5=ctrl.Rule((ang['z'] & vel['pg']) | (ang['pg'] & vel['z']),velpt['pg'])
#Decision
vel_ctrl=ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5])
vel_fn=ctrl.ControlSystemSimulation(vel_ctrl)
#Valores a evaluar
vel_fn.input['Angulo']= 3.75
vel_fn.input['Velocidad Angular']= -0.3
#Pensando
vel_fn.compute()
#Respuesta
print(vel_fn.output['Velocidad'])
velpt.view(sim=vel_fn)