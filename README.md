# gym-racecar
This is a miniature race car gym-env for RL from states (and images)

The Env is based on a real experimental setup using 1:43 scale cars

There are 4 versions of the Env
- RaceCar-v0: The car should learn racing around the race track, and uses the ideal line as a reference
- RaceCarRealistic-v0: Is identical to RaceCar-v0 but realistic noise is added to the simulation model
- RaceCarHard-v0: The agent does not know the ideal line but only the middle line
- RaceCarHardRelistic-v0: Idnetical to RaceCarHard-v0, but with noise added to the simulation

[![ORCA Platform](https://img.youtube.com/vi/JoHfJ6LEKVo/0.jpg)](https://www.youtube.com/watch?v=JoHfJ6LEKVo)
