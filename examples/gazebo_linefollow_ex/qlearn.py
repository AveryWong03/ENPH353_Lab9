import random
import pickle
import numpy as np
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        with open(filename, 'rb') as file:
            self.q = pickle.load(file)

        # TODO: Implement loading Q values from pickle file.

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename + ".pickle",'wb') as file:
            pickle.dump(self.q,file)

        Q_csv = []
        for state in self.q:
            Q_csv.append([state, self.q.get(state)])
    
        with open(filename + ".csv",'w',newline = '') as file:
            writer = csv.writer(file)
            writer.writerows(Q_csv)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''

        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        rand = random.random()

        if(rand < self.epsilon):
            randaction = random.choice(self.actions)
            if(return_q):
                return (randaction,self.getQ(state,self.actions[randaction]))
            else:
                return randaction
        else:
            Q_vals = []
            for act in self.actions:
                Q_vals.append(self.getQ(state,act))
            
            action_max = self.actions[np.argmax(Q_vals)]
            if(return_q):
                return (action_max,self.getQ(state,self.actions[action_max]))
            else:
                # If two actions have the same Q, it will default to the action with lower index (0,1,2) 
                return action_max
            

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        # if [state,action] does not exist, just dont update Q

        def checkQ(self,state,action):
            if (state,action) in self.q:
                return True
            else:
                return False
            
        if(checkQ(self,state1,action1) == False):
            # Add a dictionary entry and set to 0
            self.q[state1,action1] = 0
           
        Q_cur = self.getQ(state1,action1)

        Q2s = []
        for act in self.actions:

            if(checkQ(self,state2,act) == False):
                self.q[state2, act] = 0

            Q2s.append(self.getQ(state2, act))
        Q2max = max(Q2s)
        
        self.q[(state1,action1)] += self.alpha*(reward + self.gamma*Q2max - Q_cur)


