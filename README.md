# **Elastic Band Planner with Object detection**

### Team Member and roles

| Name | Roles |
|-------------|---------|
| Pavaris Asawakijtananont | Planner, evaluation |
| Anuwit Intet | Object Dectection |
| Bhumipat Ngamphueak | Experiment setup, evaluation |

### Table of Contents
1. Introductions
2. Project's Scope
3. Elastic Band Planner
4. Human Detection
5. Environment Setup
6. Experiment Design
7. Results 
8. Discussion
  
## 1. Introduction

Since in real environment robot have met a lot of uncertainty such as disturbance for control system and planning, since it effect to robot path can be changed inefficeint way. In social robotics, tranditional planner lack in efficient to use the information from the social such as position of human, or tracking  <please Claude will add this>

<div align="center">
  <img src="./figures/cpteb_2.png" alt="App Screenshot" width="400">
  <p><em> The out comes inthefirst andsecondrowscorrespondtotheTEBmodel andtheCPTEBmodel.Thefigures illustrate the trajectoriesofboth
therobotandthehumans.Therobot isrepresentedas thebluecirclewithtwowheels.Twopersonsaredenotedbybluecircles.Thecollisionzonesare
markedbyredcircles;andtheheadingdirectionof robotandpersonsareindicatedbythebluearrows..</em></p>
</div>

*ref : The Collision Prediction Time Elastic Band model*



## 2. Project Scope
In our project we'll improve the traditional Elastic Band planner for using the camera information for predict the information from environment for use as planning , our scope is below

1. Implement Elastic Band Local Planner
2. Detect the Human velocity with YoLO
3. Using the predicted velocity to use in planner for enchance dynamic obstacle local planner
4. Robot in this project we use robot with Holonomic constrain


### Methodology
- <How we integrated GMM from human detection and Elastic Band planner together>

## 3. Elastic Band Planner

### Multiple Convex Hull for object 

### Pre Routing
- <Why need prerouting>

## 4. Human Detection

<!-- ## 5. Environment Setup -->


## 6. Experiment Design
In experiment, we'll test our algorithm with our planner module and perception module, then we integrate both module in integration part 

**Environment Setup**
- robot velocity
- human velocity
- obstacle

### Perception
In this experiment we design to validation the accuracy of our prediction model. We employ 2 scenario for testing, by both 2 experiment robot is not moved and we vary distance between robot and human

1. Human stand in front of robot

2. Human walk pass the robot

<div align="center">
  <img src="./figures/percep_case.png" alt="App Screenshot" width="400">
  <!-- <p><em> <em></p> -->
</div>


### Planner and Integration
In this experiment we setting up 3 scenario for testing our plnner algorith, included with Static Obstacle, Dynamic obstacle with human walk from same side, and from opposite side

<div align="center">
  <img src="./figures/planner_case.png" alt="App Screenshot" width="400">
  <!-- <p><em> <em></p> -->
</div>

1. **Static Obstacle** :  This scenario contain with 2 obstacle to blocking the initial trajectory between initial robot pose and goal 


2. **Dynamic Obstalce (Same)** : <Claude look at ./figures/planner_case.png in case C and describe>

3. **Dynamic Obstacle (Opposite)** : <Claude look at ./figures/planner_case.png in case C and describe>

## 7. Results 
[!note] All result videos are contains in `videos` folder

### Perception


### Planner & System Integration

| World | Look ahead | Path distance (m) | Reach Goal |
| :--- | :---: | :---: | :---: |
| **2 Static Obstacles (B)** | 2 | 9.9 | No |
| | 5 | 16.9 | Yes |
| | 10 | 16.8 | Yes |
| | 15 | 17 | Yes |
| **2 Static Obstacles (B) no_preroute** | 5 | 10.2 | No |
| **2 human with opposite directions (C)** | 2 | 15.1 | Yes |
| | 5 | 17 | Yes |
| | 10 | 18.2 | Yes |
| **2 human with same directions (D)** | 2 | 15.7 | Yes |
| | 5 | 16.5 | Yes |
| | 10 | 16.6 | Yes |



#### **Static Obstacle (Planner Only)**
  - It can beseen that without prerouting, robot cannot acheive the goal, while when at the lowest lookahead distance at 2 m robot cannot pass through the obstacle as the same as wthout prerouting
  - Moreover, We can observe that, when we increase the lookahead distance of planner which made the distance increasing when increase the lookahead distance

<table>
  <tr>
    <td>
      <video src="./videos/world2_look_2.webm" width="100%" controls></video>
      <p align="center">Caption for Video 1</p>
    </td>
    <td>
      <video src="./videos/world2_look_5.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
    </td>
        <td>
      <video src="./videos/world2_look_10.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <video src="./videos/world2_look_15.webm" width="100%" controls></video>
      <p align="center">Caption for Video 1</p>
    </td>
    <td>
      <video src="./videos/world2_no_preroute_look5.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="./figures/w2_p.png" width="400" alt="Figure 1">
      <p>Caption for Figure 1</p>
    </td>
  </tr>
  
  <tr>
    <td align="center">
      <img src="./figures/w3_p.png" width="400" alt="Figure 2">
      <p>Caption for Figure 2</p>
    </td>
  </tr>
  
  <tr>
    <td align="center">
      <img src="./figures/w4_p.png" width="400" alt="Figure 3">
      <p>Caption for Figure 3</p>
    </td>
  </tr>
</table>


#### **Dynamic Obstacle (Planner Only)**
  -  In dynamic obstacle environment we can observe with any lookahead distance, our robot can pass both human and achieve the goal but, the robot is fail to avoid the collision of first human,
  - We also can observe that when lookahead is increasing, our distance also higher since our robot plan for longer horizon, it change the further node quickly than it interact that make the path is more longer.
  - The path can also shift it Homotopy(after pass the human 1), by human 1 cause contraction force is more than repulsive force from human

##### World C
<table>
  <tr>
    <td>
      <video src="./videos/w3__delay6_look2.webm" width="100%" controls></video>
      <p align="center">Caption for Video 1</p>
    </td>
    <td>
      <video src="./videos/w3__delay6_look5.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
    </td>
        <td>
      <video src="./videos/w3__delay6_look10.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
    </td>
  </tr>
</table>

##### World D
<table>
  <tr>
    <td>
      <video src="./videos/world4_look2.webm" width="100%" controls></video>
      <p align="center">Caption for Video 1</p>
    </td>
    <td>
      <video src="./videos/world4_look5.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
    </td>
        <td>
      <video src="./videos/world4_look10.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
    </td>
  </tr>
</table>

##### Free Environment

<table>
  <tr>
    <td>
      <video src="./videos/Toy_Exp1_look_2.webm" width="100%" controls></video>
      <p align="center">Testing integration of GMM and EB planner with lookahead 2 m</p>
    </td>
    <td>
      <video src="./videos/Toy_Exp1_look_5.webm" width="100%" controls></video>
      <p align="center">Testing integration of GMM and EB planner with lookahead 5 m</p>
    </td>
        <td>
      <video src="./videos/Toy_Exp1_look_10.webm" width="100%" controls></video>
      <p align="center">Testing integration of GMM and EB planner with lookahead 10 m</p>
    </td>
  </tr>
</table>

from the free environment it significantly see that robot path can change depending on lookahead if lookahead is too high path is can more easier change depend on environment and too fast change in future node it can make path not efficient

#### **Dynamic Obstacle (Integrated System)**

##### World C,D
<!-- <table>
  <tr>
    <td>
      <video src="./videos/world3_egg_look5.webm" width="100%" controls></video>
      <p align="center">Caption for Video 1</p>
    </td>
    <td>
      <video src="./videos/world4_egg_look5.webm" width="100%" controls></video>
      <p align="center">Caption for Video 2</p>
</table> -->




<div align="center">
  <img src="./figures/GMM_res.png" alt="App Screenshot" width="400">
  <p><em> <em></p>
</div>

Since, our system after integrated robot still collision the first human so we tested on the free environment. So the key main reason is the FOV of camera and direction of robot not face to human that make we cannot exploit the benefit of using GMM for use as information for planning




##  8. Discussion
Our framework success to avoid the static unseen collision ,but fail to make the robot avoid the collision and preventing the problem of dynamic obstacle.

1) **Robot Height** — Low Camera Loses Human
Camera at ~1.0 m → at 1–2 m range, only legs visible → YOLO drops detection
Social costmap vanishes → planner thinks path is clear → drives into human
Reference paper uses tall service robot (camera ~1.4–1.6 m) → sees torso/face even at close range

2) **Limited FOV** — Human Exits View
60° camera cone → human only visible within ±30° ahead
During avoidance maneuver, robot turns away → human leaves FOV → detection lost
Loses obstacle info precisely when it's needed most

3) **Situation Mismatch** — Open Space vs. Hotel Hallway
Reference paper operates in narrow hotel corridors (~2 m wide)
Robot avoids sideways but hallway walls keep human within FOV → detection maintained → smooth avoidance
Our simulation uses wide open environments
Elastic band swings the robot far to the side → human exits FOV entirely → social costmap disappears → planner overcorrects or collides
Narrow space is actually an advantage for this perception setup — the geometry constrains both the robot and human to stay visible to each other

Root Cause: All three issues compound at the same moment — the robot is close, turning away, in open space — creating a detection blackout exactly when the planner needs it most.

4) **Experiment setting up** — Other reason is experiment setting, like relative velocity between robot and human, which effect to interaction time, if robot too slow robot cannot pass the front human, but if robot too fast it do not care about human path much.

