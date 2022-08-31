#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd
torch.multiprocessing.set_start_method('forkserver', force=True)
from torch.multiprocessing import Process, Value, Manager, Queue, Lock
import random
import sys
import csv
import math
import statistics
import numpy as np
import itertools
import copy
import time
from Instances import *


import argparse

parser = argparse.ArgumentParser(description='Arguments of Experiments')
parser.add_argument("--eps", default=350000, type=int, help="Episodes")
parser.add_argument("--steps", default=600, type=int, help="Steps")
parser.add_argument("--lr", default=0.00004, type=float, help="Learning Rate (at the start)")
parser.add_argument("--df", default=0.9, type=float, help="Discount Factor")
parser.add_argument("--rb", default=40, type=int, help="Replay Buffer size in episodes")
parser.add_argument("--bs", default=500, type=int, help="Batch size")
parser.add_argument("--epochs", default=5, type=int, help="Epochs")
parser.add_argument("--hu1", default=64, type=int, help="Hidden units for layer 1")
parser.add_argument("--hu2", default=32, type=int, help="Hidden units for layer 2")
parser.add_argument("--instance", default=0, type=int, help="Instance Number")
parser.add_argument("--arc", default=0.000006, type=float, help="Exploration Curve decay")
parser.add_argument("--n", default=0.9, type=float, help="Exploration Curve start")
parser.add_argument("--agents", default=10, type=int, help="Total Agents")
parser.add_argument("--grid", default=4, type=int, help="Grid size")
# parser.add_argument("--log", default=10000, type=int, help="Log interval of agent trails")
parser.add_argument("--minER", default=0.05, type=float, help="Minimum exploration rate")
parser.add_argument("--decay", default=0.999996, type=float, help="Learning rate decay factor per step")
parser.add_argument("--minR", default=-1, type=float, help="Reward for NOT being in a match")
parser.add_argument("--prefType", default='asym', type=str, help="Preference Type: Symmetric ('sym') or Asymmetric")
parser.add_argument("--problemType", default='sm', type=str, help="Problem Type: sm, smi or smt")



# parser.add_argument("--folder", required=True, type=str, help="Data Folder")

args = parser.parse_args()

# data_folder= args.folder

if args.problemType=='sm':
    if args.prefType=='asym':
        if args.agents==8:
            preferences= smAsymInstances8[args.instance]
        elif args.agents==10:
            preferences= smAsymInstances10[args.instance]
        elif args.agents==12:
            preferences= smAsymInstances12[args.instance]
        elif args.agents==14:
            preferences= smAsymInstances14[args.instance]
    elif args.prefType=='sym':
        if args.agents==8:
            preferences= smSymInstances8[args.instance]
        elif args.agents==10:
            preferences= smSymInstances10[args.instance]
        elif args.agents==12:
            preferences= smSymInstances12[args.instance]
        elif args.agents==14:
            preferences= smSymInstances14[args.instance]        
    else:
        print('Wrong preference type!')
        
elif args.problemType=='smi':
    if args.prefType=='asym':
        if args.agents==8:
            preferences= smiAsymInstances8[args.instance]
        elif args.agents==10:
            preferences= smiAsymInstances10[args.instance]
        elif args.agents==12:
            preferences= smiAsymInstances12[args.instance]
        elif args.agents==14:
            preferences= smiAsymInstances14[args.instance]        

    elif args.prefType=='sym':
        if args.agents==8:
            preferences= smiSymInstances8[args.instance]
        elif args.agents==10:
            preferences= smiSymInstances10[args.instance]
        elif args.agents==12:
            preferences= smiSymInstances12[args.instance]
        elif args.agents==14:
            preferences= smiSymInstances14[args.instance]        
    else:
        print('Wrong preference type!')
    
elif args.problemType=='smt':
    if args.prefType=='asym':
        if args.agents==8:
            preferences= smtAsymInstances8[args.instance]
        elif args.agents==10:
            preferences= smtAsymInstances10[args.instance]
        elif args.agents==12:
            preferences= smtAsymInstances12[args.instance]
        elif args.agents==14:
            preferences= smtAsymInstances14[args.instance]        

    elif args.prefType=='sym':
        if args.agents==8:
            preferences= smtSymInstances8[args.instance]
        elif args.agents==10:
            preferences= smtSymInstances10[args.instance]
        elif args.agents==12:
            preferences= smtSymInstances12[args.instance]
        elif args.agents==14:
            preferences= smtSymInstances14[args.instance]        
    else:
        print('Wrong preference type!')
    
else:
    print('Wrong problem type!')
    
    

expR=[]

n=args.n
episodes=args.eps
stepsInEpisode=args.steps
for i in range(episodes+1):
    n+=args.arc
    expR.append(math.exp(-n))

min_exploration_rate = args.minER

discount_factor=args.df
learning_rate=args.lr
lr_decay = args.decay

grid1= args.grid
grid2= args.grid
totalAgents= args.agents
gridStates = grid1 * grid2
mainState = gridStates + 2 * int(totalAgents/2)
mainOutput = 4 + int(totalAgents/2)

hidden_units1=args.hu1
hidden_units2=args.hu2

batch_size=args.bs
bufferEps=args.rb
epochs=args.epochs



def update(model, MSELoss, optimizer, replay_buffer, episodeCounter2):
    for i in range(epochs):
        sampling= random.sample(replay_buffer, batch_size)
        state=[item[0] for item in sampling]
        action=[item[1] for item in sampling]
        reward=[item[2] for item in sampling]
        next_state=[item[3] for item in sampling]
        next_state_actions=[item[4] for item in sampling]
        
        states = torch.FloatTensor(state)
        actions = torch.LongTensor(action)
        rewards = torch.FloatTensor(reward)
        next_states = torch.FloatTensor(next_state)

        curr_Q = model(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = model(next_states)
        
        m=[]
        
        actions_basic=[0,1,2,3]
        it=0
        for each in next_Q:
            z=torch.max(each,0).values
            exploration_rate=expR[episodeCounter2.value]
            if exploration_rate < min_exploration_rate:
                exploration_rate= min_exploration_rate
            r=random.random()
            if r<=exploration_rate:
                act=random.choice(actions_basic+next_state_actions[it])
                z=each[act]
            m.append(z)
            it+=1
            
        m=torch.stack(m)
        expected_Q = rewards + discount_factor * m
        loss = MSELoss(curr_Q, expected_Q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Agent():
    def __init__(self, unique_id, position, grid1, grid2, totalAgents, preferences,
                replay_buffer, allMales, allFemales, rewardList, cumulativeRewardList, testModeRewards, testModeCumulativeRewards,
                proposed, reward, cumulativeReward, action,
                discount_factor,inMatch,avgTestModeR, avgTestModeCR, state, next_state, simplifyOutput, simplifyAction, 
                 gridStates, expR, mainOutput):
        self.unique_id=unique_id
        self.pos=position
        self.grid1= grid1
        self.grid2= grid2
        self.preferences=preferences
        self.totalAgents=totalAgents
        self.state=state
        self.next_state=next_state
        self.proposed= proposed
        self.reward=reward
        self.cumulativeReward=cumulativeReward
        self.action=action
        self.simplifyOutput=simplifyOutput
        self.simplifyAction=simplifyAction
        self.expR=expR
        self.discount_factor=discount_factor
        self.gridStates=gridStates
        self.mainOutput=mainOutput
        
        self.inMatch=inMatch

        self.avgTestModeR =avgTestModeR
        self.avgTestModeCR =avgTestModeCR
        
        self.replay_buffer=replay_buffer
        self.allMales=allMales
        self.allFemales=allFemales
        self.rewardList=rewardList
        self.cumulativeRewardList=cumulativeRewardList
        self.testModeRewards=testModeRewards
        self.testModeCumulativeRewards=testModeCumulativeRewards

def moveX(obj,direction):
    if direction==1:
        if obj.pos[1]>0:
            obj.pos=(obj.pos[0],obj.pos[1]-1)
    elif direction==2:
        if obj.pos[1]<(obj.grid2-1):
            obj.pos=(obj.pos[0],obj.pos[1]+1)
    elif direction==3:
        if obj.pos[0]>0:
            obj.pos=(obj.pos[0]-1,obj.pos[1])
    elif direction==4:
        if obj.pos[0]<(obj.grid1-1):
            obj.pos=(obj.pos[0]+1,obj.pos[1])


def learningX(obj, cellmates, testMode, proposals, episodeCounter, QNetwork):
    cellmates.remove(obj.unique_id)
    cellmates_male=[]
    cellmates_female=[]
    actions_choice=[0,1,2,3]
    
    obj.state=np.zeros((1,mainState), dtype=float)
    obj.state[0][obj.pos[0]*obj.grid2+obj.pos[1]]=1.0

    if len(cellmates)>1:
        newActions=[]
        if obj.unique_id in obj.allMales:
            for each in cellmates:
                if each in obj.allFemales:
                    cellmates_female.append(each)
                    newActions.append(obj.simplifyAction[each])

            if cellmates_female:
                for each in cellmates_female:
                    obj.state[0][obj.gridStates + each - int(obj.totalAgents/2)]=1.0
                    if each in proposals:
                        obj.state[0][obj.gridStates + int(obj.totalAgents/2) + each - int(obj.totalAgents/2)]=2.0

        elif obj.unique_id in obj.allFemales:
            for each in cellmates:
                if each in obj.allMales:
                    cellmates_male.append(each)
                    newActions.append(obj.simplifyAction[each])
                    
            if cellmates_male:
                for each in cellmates_male:
                    obj.state[0][obj.gridStates + each]=1.0
                    if each in proposals:
                        obj.state[0][obj.gridStates + int(obj.totalAgents/2) + each]=2.0
                        
        actions_choice += newActions
        
    obj.reward=args.minR
    obj.inMatch=None
    obj.proposed= None    
    
    temp_state=torch.from_numpy(obj.state).float().flatten()

    QValues = QNetwork(temp_state)                    

    _, tempaction = torch.max(QValues,0)
    obj.action=tempaction.item()

    
    exploration_rate=obj.expR[episodeCounter]
    if exploration_rate < min_exploration_rate:
        exploration_rate= min_exploration_rate
    r=random.random()
    if r<=exploration_rate and testMode==False:
        obj.action=random.choice(actions_choice)

    if obj.action==0:
        moveX(obj, 1)
    elif obj.action==1:
        moveX(obj, 2)
    elif obj.action==2:
        moveX(obj, 3)
    elif obj.action==3:
        moveX(obj, 4)
    else:
        moveX(obj, 0)
        if len(cellmates)>1:
            if obj.unique_id in obj.allMales:
                if obj.action in range(4,4+int(obj.totalAgents/2)):
                    f=obj.simplifyOutput[obj.action]
                    if f in cellmates_female:
                        obj.proposed= f

            elif obj.unique_id in obj.allFemales:
                if obj.action in range(4,4+int(obj.totalAgents/2)):
                    m=obj.simplifyOutput[obj.action]
                    if m in cellmates_male:
                        obj.proposed= m


def advanceX(obj,cellmates,testMode, proposals):
    obj.next_state=np.zeros((1,mainState), dtype=float)
    obj.next_state[0][obj.pos[0]*obj.grid2+obj.pos[1]]=1.0

    flag=False
    
    cellmates.remove(obj.unique_id)
    cellmates_male=[]
    cellmates_female=[]
    
    next_state_actions=[]

    if len(cellmates)>1:
        if obj.unique_id in obj.allMales:
            for each in cellmates:
                if each in obj.allFemales:
                    cellmates_female.append(each)
            if cellmates_female:
                for each in cellmates_female:
                    obj.next_state[0][obj.gridStates + each - int(obj.totalAgents/2)]=1.0
                    next_state_actions.append(obj.simplifyAction[each])
                    if each in proposals:
                        obj.next_state[0][obj.gridStates + int(obj.totalAgents/2) + each - int(obj.totalAgents/2)]=2.0
                        if obj.proposed==each:
                            flag=True

        elif obj.unique_id in obj.allFemales:
            for each in cellmates:
                if each in obj.allMales:
                    cellmates_male.append(each)
            if cellmates_male:
                for each in cellmates_male:
                    obj.next_state[0][obj.gridStates + each]=1.0
                    next_state_actions.append(obj.simplifyAction[each])
                    if each in proposals:
                        obj.next_state[0][obj.gridStates + int(obj.totalAgents/2) + each]=2.0
                        if obj.proposed==each:
                            flag=True
    if flag==True:
        obj.inMatch=obj.proposed
        noise=np.random.normal(loc=1,scale=0.1)
        obj.reward= noise*obj.preferences[obj.unique_id][obj.proposed]

    obj.cumulativeReward+=obj.reward
    if testMode==False:
        obj.replay_buffer.append((obj.state.flatten(), obj.action, obj.reward, obj.next_state.flatten(), next_state_actions))
        



def trainingInBackground(episodeCounter2, rb_queue, pc_queue, lock, lock_pc, thread_id):
    QNetwork= torch.nn.Sequential(torch.nn.Linear(mainState,hidden_units1),torch.nn.ReLU(),
                                 torch.nn.Linear(hidden_units1,hidden_units2),torch.nn.ReLU(),
                                 torch.nn.Linear(hidden_units2,mainOutput),torch.nn.ReLU())
    MSELoss= torch.nn.MSELoss()
    optimizer = torch.optim.Adam(QNetwork.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    rb=[]
    fitCount=0
    prev_ep=0
    count=0
    while episodeCounter2.value < episodes:
        if len(rb)>= batch_size/stepsInEpisode:
            flat_rb= [x for l in rb for x in l]
            temp_rb=copy.copy(flat_rb)
            update(QNetwork, MSELoss, optimizer, temp_rb, episodeCounter2)
            scheduler.step()
            fitCount+=1
        if episodeCounter2.value>prev_ep:
            lock_pc.acquire()
            pc_queue.put(QNetwork.state_dict())
            lock_pc.release()
            prev_ep=episodeCounter2.value
        if not rb_queue.empty():
            lock.acquire()
            qitem=rb_queue.get()
            lock.release()
            if len(rb)>=bufferEps:
                rb[count]=qitem
            else:
                rb= rb + [qitem]          
            count+=1                                
            if count>bufferEps-1:
                count=0
    print('FIT Count: ',fitCount)
    sys.stdout.flush()

QNetworks={}
for agentIndex in range(totalAgents):
    QNetworks[agentIndex]= torch.nn.Sequential(torch.nn.Linear(mainState,hidden_units1),torch.nn.ReLU(),
                                 torch.nn.Linear(hidden_units1,hidden_units2),torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_units2,mainOutput),torch.nn.ReLU())

            
if __name__ == '__main__':
    episodeCounter2= Value('i',0)

    testModeFrq=20
    testModeAvgFrq=10
    
    originalgrid=[]
    for i in range(grid1):
        temp=[]
        for j in range(grid2):
            temp2=[]
            temp.append(temp2)
        originalgrid.append(temp)
#     print(originalgrid)
    
    originalpos={}
    for a in range(totalAgents):
        i=random.randrange(0,grid1)
        j=random.randrange(0,grid2)
        originalgrid[i][j].append(a)
        originalpos[a]=(i,j)

    allMales=[]
    for m in range(totalAgents//2):
        allMales.append(m)
    allFemales=[]
    for f in range(totalAgents//2, totalAgents):
        allFemales.append(f)
#     print(allMales,allFemales)


    objList=[]
    objDict={}
    
    replay_buffer_queue= {}
    param_copy_queue= {}
    locks= {}
    locks_pc= {}
    
    proposals_dict={}
    matched_dict={}
    cellmates_dict={}
#     trails={}


    for a in range(totalAgents):
        simplifyOutput= {}
        simplifyAction= {}
        if a in allMales:
            j=0
            for i in range(4,4 + int(totalAgents/2)):
                simplifyOutput[i]=allFemales[j]
                simplifyAction[allFemales[j]]= i
                j+=1
        if a in allFemales:
            j=0
            for i in range(4,4 + int(totalAgents/2)):
                simplifyOutput[i]=allMales[j]
                simplifyAction[allMales[j]]= i
                j+=1

        rewardList=[]
        cumulativeRewardList=[]
        testModeRewards=[]
        testModeCumulativeRewards=[]
        replay_buffer=[]
        
        proposed= 100
        reward=args.minR
        cumulativeReward=0.0
        action=None

        inMatch=None

        testMode=False

        avgTestModeR =0
        avgTestModeCR =0

        state=np.zeros((1,mainState), dtype=float)
        next_state=np.zeros((1,mainState), dtype=float)

        
        obj = Agent(a, originalpos[a], grid1, grid2, totalAgents, preferences,
                   replay_buffer, allMales, allFemales, rewardList, cumulativeRewardList, testModeRewards,
                    testModeCumulativeRewards, proposed, reward, cumulativeReward, action,
                    discount_factor,inMatch,avgTestModeR, avgTestModeCR, state, next_state, simplifyOutput, simplifyAction,
                    gridStates, expR, mainOutput)
        objList.append(obj)
        objDict[a]=obj
        replay_buffer_queue[a]= Queue()
        param_copy_queue[a]= Queue()
        locks[a]= Lock()
        locks_pc[a]= Lock()
        
        proposals_dict[a]= []
        matched_dict[a]= None
        cellmates_lst=[]
        cellmates_dict[a]=cellmates_lst
#         trails[a]=[]


    tmCount=0
    tmEps=[]
    totalRewardsList=[]
    totalCumuRewardsList=[]
    testModeTotalRewardList= []
    testModeTotalCumuRewardList= []
    
        
    threads_bg=[]
    for obj in objList:
        t = Process(target=trainingInBackground, args=(episodeCounter2, replay_buffer_queue[obj.unique_id], 
                                                       param_copy_queue[obj.unique_id], locks[obj.unique_id],
                                                       locks_pc[obj.unique_id], obj.unique_id))
        threads_bg.append(t)
        t.start()

    
    for key in proposals_dict.keys():
        proposals_dict[key].append(333)
        del proposals_dict[key][:]
        
    cal_time=time.time()
    
    for e in range(episodes):
        episodeCounter2.value=e
        testMode=False
        if e % testModeFrq==0 and e>0:
            testMode=True
            tmCount+=1

        for agentNumber in range(totalAgents):
            if not param_copy_queue[agentNumber].empty():
                locks_pc[agentNumber].acquire()
                QNetworks[agentNumber].load_state_dict(param_copy_queue[agentNumber].get())
                locks_pc[agentNumber].release()
            
        cells={}
        for obj in objList:
            obj.pos= originalpos[obj.unique_id]
            if obj.pos not in cells.keys():
                cells.update({obj.pos:[]})
            cells[obj.pos].append(obj.unique_id)
            
#         if e % args.log==0:
#             for obj in objList:
#                 trails[obj.unique_id].append(obj.pos)


        for step in range(stepsInEpisode):                    
            for key in proposals_dict.keys():
                del proposals_dict[key][:]
                
            for key in matched_dict.keys():
                matched_dict[key]=None

            for obj in objList:
                cellmates=copy.deepcopy(cells[obj.pos])
                learningX(objDict[obj.unique_id], cellmates, testMode, proposals_dict[obj.unique_id], e, QNetworks[obj.unique_id])
                if obj.proposed!=None:
                    proposals_dict[obj.proposed].append(obj.unique_id)
        
            cells={}
            for obj in objList:
                if obj.pos not in cells.keys():
                    cells.update({obj.pos:[]})
                cells[obj.pos].append(obj.unique_id)
            
            for obj in objList:
                cellmates=copy.deepcopy(cells[obj.pos])
                advanceX(obj,cellmates,testMode, proposals_dict[obj.unique_id])
                if obj.inMatch!=None:
                    matched_dict[obj.inMatch]= obj.unique_id
            for obj in objList:
                if matched_dict[obj.unique_id]!=None:
                    obj.inMatch= matched_dict[obj.unique_id]
                    noise=np.random.normal(loc=1,scale=0.1)
                    obj.reward= noise*obj.preferences[obj.unique_id][obj.inMatch]
#             if e % args.log==0:
#                 for obj in objList:
#                     trails[obj.unique_id].append(obj.pos)
                
                
    
        add=0
        cumuAdd=0
        tmAdd=0
        tmCumuAdd=0
        displayCount=0
        for obj in objList:
            if testMode==False:
                locks[obj.unique_id].acquire()
                replay_buffer_queue[obj.unique_id].put(list(obj.replay_buffer))
                locks[obj.unique_id].release()
            del obj.replay_buffer[:]
            
#             if e % args.log==0:
#                 with open('/project/'+data_folder+'/agentsTrails.csv', 'a') as csv_file:
#                     writer = csv.writer(csv_file)
#                     writer.writerow(trails[obj.unique_id])
#             trails[obj.unique_id]=[]

            if obj.inMatch != None:
                print('ID: ',obj.unique_id, 'Reward: ', round(obj.reward,1), ' Match: ',obj.inMatch)
                displayCount+=1
            add+= obj.reward
            cumuAdd+= obj.cumulativeReward
            obj.rewardList.append(obj.reward)
            obj.cumulativeRewardList.append(obj.cumulativeReward)

            if testMode==True:
                obj.avgTestModeR +=obj.reward
                obj.avgTestModeCR +=obj.cumulativeReward
                if tmCount==testModeAvgFrq:
                    avgCR= obj.avgTestModeCR/testModeAvgFrq
                    avgR= obj.avgTestModeR/testModeAvgFrq
                    tmCumuAdd += avgCR
                    tmAdd +=  avgR
                    obj.testModeRewards.append(avgR)
                    obj.testModeCumulativeRewards.append(avgCR)
                    obj.avgTestModeR = 0
                    obj.avgTestModeCR = 0
#         if e % args.log==0:
#             with open('/project/'+data_folder+'/agentsTrails.csv', 'a') as csv_file:
#                 writer = csv.writer(csv_file)
#                 writer.writerow([])


        if tmCount==testModeAvgFrq:
            tmEps.append(e)
            testModeTotalRewardList.append(tmAdd)
            testModeTotalCumuRewardList.append(tmCumuAdd)
            tmCount=0
        if add>0:
            if testMode==True:
                print('------------TEST MODE-------------')
            print('Episode: ',e, ' Total Reward: ',round(add,2))
            if displayCount==totalAgents:
                print('\n','**********'*20)
        sys.stdout.flush()
        totalRewardsList.append(round(add,2))
        totalCumuRewardsList.append(round(cumuAdd,2))
        
    episodeCounter2.value+=1
    for each in threads_bg:
        each.join()
    print('DONE!!! Total Time: ',time.time()-cal_time)
#     for obj in objList:
#         tempList= list(obj.rewardList)
#         df = pd.DataFrame(tempList)
#         df.to_csv('/project/'+data_folder+'/Agent'+str(obj.unique_id)+'_Rewards.csv', index=False)
        
#         tempList= list(obj.cumulativeRewardList)
#         df = pd.DataFrame(tempList)
#         df.to_csv('/project/'+data_folder+'/Agent'+str(obj.unique_id)+'_CumuRewards.csv', index=False)
        
#         tempList= list(obj.testModeRewards)
#         df = pd.DataFrame(tempList)
#         df.to_csv('/project/'+data_folder+'/Agent'+str(obj.unique_id)+'_TestMode_Rewards.csv', index=False)
        
#         tempList= list(obj.testModeCumulativeRewards)
#         df = pd.DataFrame(tempList)
#         df.to_csv('/project/'+data_folder+'/Agent'+str(obj.unique_id)+'_TestMode_CumuRewards.csv', index=False)

#     tempList= list(totalRewardsList)
#     df = pd.DataFrame(tempList)
#     df.to_csv('/project/'+data_folder+'/Total_Rewards.csv', index=False)

#     tempList= list(totalCumuRewardsList)    
#     df = pd.DataFrame(tempList)
#     df.to_csv('/project/'+data_folder+'/Total_CumuRewards.csv', index=False)
    
#     tempList= list(testModeTotalRewardList)    
#     df = pd.DataFrame(tempList)
#     df.to_csv('/project/'+data_folder+'/TestMode_Total_Rewards.csv', index=False)
    
#     tempList= list(testModeTotalCumuRewardList)    
#     df = pd.DataFrame(tempList)
#     df.to_csv('/project/'+data_folder+'/TestMode_Total_CumuRewards.csv', index=False)
    
#     print('FILES WRITTEN!!!')
    sys.stdout.flush()







