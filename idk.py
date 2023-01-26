import ludopy
class env:
    def __init__(self, rang):
        self.rang = rang
    def get_state(self, ):
        pass
    def get_rewrad(self,):
        ()
    def step(self, action):
        tot_reward = 0.0
        done = False
        for i in range(self.rang):
            obs, reward, done, info = self.step(action)
            tot_reward += reward
            if done:
                break
        return obs, tot_reward, done, info