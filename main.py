# 바닥부터 배우는 강화 학습 P.146 SARSA 구현

import random
import numpy as np

class GridWorld(): # MC Control과 동일
    def __init__(self):
        self.x = 0
        self.y = 0
        self.first_block = range(3)
        self.second_block = range(2, 5)
    
    # Agent의 움직임을 나타냄
    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()
        
        state = self.get_state()
        reward = -1
        done = self.is_done()

        return state, reward, done
    
    # 회색인 부분으로는 갈 수 없음
    def move_left(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y in self.first_block:
            pass
        elif self.x == 5 and self.y in self.second_block:
            pass
        else:
            self.x -= 1
    
    def move_right(self):
        if self.x == 6:
            pass
        elif self.x == 1 and self.y in self.first_block:
            pass
        elif self.x == 3 and self.y in self.second_block:
            pass
        else:
            self.x += 1
    
    # y축 방향이 반대임
    def move_up(self):
        if self.y == 0:
            pass
        elif self.x == 2 and self.y == 3:
            pass
        else:
            self.y -= 1
    
    def move_down(self):
        if self.y == 4:
            pass
        elif self.x == 4 and self.y == 1:
            pass
        else:
            self.y += 1

    # 종료 State에 도달했는지 체크
    def is_done(self):
        if self.x == 6 and self.y == 4:
            return True
        else:
            return False
    
    # 현재 Agent가 위치한 State를 반환
    def get_state(self):
        return (self.x, self.y)
    
    # 종료 State에 도달했을 때 리셋
    def reset(self):
        self.x = 0
        self.y = 0

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((7, 5, 4))
        # epsilon, decaying이므로 처음엔 큰 수를 넣음
        self.eps = 0.9 

    # decaying epsilon-greedy에 따라 Action을 결정함
    def select_action(self, state): 
        x, y = state
        coin = random.random()
        # 랜덤 Action을 선택하는 경우
        if coin < self.eps: 
            action = random.randint(0, 3)
        else:
            # 현 State의 Action Value를 모두 불러옴
            action_val = self.q_table[x, y, :]
            # Action Value 중 가장 높은 것을 선택
            action = np.argmax(action_val)
        return action
    
    def update_table(self, transition):
        state, action, reward, state_prime = transition
        x, y = state
        x_prime, y_prime = state_prime
         # s'에서 선택할 Action, 현재 State에서 선택한 게 아님.
        action_prime = self.select_action(state_prime)
        # SARSA 업데이트 식을 이용함
        self.q_table[x, y, action] = self.q_table[x, y, action] + 0.1 * (reward + self.q_table[x_prime, y_prime, action_prime]-self.q_table[x, y, action])

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)
    
    def show_table(self): # 각 State별 가장 높은 Action Value 출력
        q_list = self.q_table.tolist()
        data = np.zeros((7,5))
        for row_idx in range(len(data)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx] # 각 State의 Action 4개
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data.T)

def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000): # 1000번의 Episode 진행
        done = False

        env.reset()
        state = env.get_state()
        while not done:
            action = agent.select_action(state)
            state_prime, reward, done = env.step(action)
            agent.update_table((state, action, reward, state_prime))
            state = state_prime
        agent.anneal_eps()

    agent.show_table()

main()