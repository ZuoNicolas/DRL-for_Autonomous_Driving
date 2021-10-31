import torch
from pathlib import Path
#savepath = Path('model/model_DQN_EnvDistanceReward_600-1635352803.4654603.pch')
savepath = Path('model/model_DQN_EnvDistanceReward_300-1635331329.0202696.pch')
with savepath.open('rb') as file:
    model = torch.load(file)


if __name__ == '__main__':

    episode = 10
    render = True
    #env = CarEnv()
    env = CarEnvDistanceReward(reward_function=reward_function,EXPERIENCE_SECONDE= 10)
    start_time = time.time()
    for i in range(episode):
        score = 0
        done = False
        observation = env.reset()
        flat_observation = observation.reshape(1,-1)[0]/255.0
        try : 
            while not done:

                if render:
                    cv2.imshow(f'Agent - preview', observation)
                    cv2.waitKey(1)

                data = T.tensor(flat_observation).float()

                action = model.forward(data)
                action = action.detach().numpy().argmax()

                observation_, reward, done, info = env.step(action)
                flat_observation_ = observation_.reshape(1,-1)[0]/255.0
                score += reward
                flat_observation = flat_observation_
                observation = observation_

        finally : 
            if render:
                cv2.destroyWindow(f'Agent - preview')
            for actor in env.actor_list:
                actor.destroy()
            time_n = time.time() - start_time
            print('episode ', i, 'score %.2f' % score,'time %.2f s' % time_n)