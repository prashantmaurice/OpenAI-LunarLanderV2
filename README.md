# LunarLander V2 using Neural Networks

A simple network that can learn to land a spacecraft gently using reinforcement learning aided by Tensorflow

OpenAI link: 
https://gym.openai.com/envs/LunarLander-v2

### Results

Results while Training

| Average Score | Episodes |
| ------ | ------ |
| -878 | 1 |
| -454 | 50 |
| -442 | 100 |
| -415 | 200 |
| -380 | 400 |
| -246 | 600 |
| -230 | 1000 |


### Issues while building
if Box2d is not available in your machine, install it via following commands
1) `cd ./universe`
2) `git clone https://github.com/pybox2d/pybox2d pybox2d_dev`
3) `cd pybox2d_dev`
4) `python setup.py build`
5) `sudo python setup.py develop`

if swig is unavailable in your machine install it via
1) `brew install swig`