from kaggle_environments import evaluate, make
from kaggle_environments.envs.hungry_geese import hungry_geese

if __name__ == "__main__":
    env = make("hungry_geese")
    env.reset()
    env.run(['submission.py', 'submission.py', 'submission.py', 'submission.py'])
    env.render(mode="ipython", width=800, height=700)