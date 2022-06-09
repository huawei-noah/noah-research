# O2DESpy

## Overview

O2DESpy is a framework for object-oriented discrete event simulation based on standard Python 3.x. It is developed by Huawei, Noah's Ark Lab and the National University of Signapore, and extended from [O2DES](https://github.com/li-haobin/O2DESNet).

It hybrids both event-based and state-based formalism, and implement them in an object-oriented programming language. As it is event-based in the kernel, O2DESpy is able to model the structure and behaviours of a system precisely. On top of it, the state-based formalism enables modularization and hierarchical modelling. Besides, the object-oriented paradigm abstracts the model definitions and makes them seamless to interact with analytical functionalities, regardless of their fidelity levels.

The framework underpinning the O2DESpy package is the Sandbox. The Sandbox module allows for the intuitive construction of components in a simulation model such as Generator, Queue & Server as Sandboxes. A unique feature of the Sandbox module is the innate ability of nesting these components (Sandboxes). This allows for a system of components to utilise a common pool of pre-defined methods as well as wrap these components into a single system of events so that the clock time, scheduled event list and sequence of these events can be in sync in a single simulation model.

The sections below contain an installation guide and dependencies to illustrate how O2DESpy can be installed and be used to build a discrete event simulation model. You can explore our package for more demo examples.

## Getting Started

#### Dependencies

Listed below are packages required by the O2DESpy package.

| Package          | Version Number |
| ---------------- | -------------- |
| abc              | 3.4            |
| sortedcontainers | 2.3.0          |

## Simulation Model Construction (Hello World!)

The first example would be to simulate sequential arrival events whereby at each arrival, the simulation would output and greet the user "Hello World!". To begin, the necessary imports and child class inheritance of the Sandbox model is shown below.

#### Imports

``` python
from O2DESpy.sandbox import Sandbox
import datetime
import random
```

#### Child Class of Sandbox

``` python
class HelloWorld(Sandbox):
    def __init__(self, hourly_arrival_rate, seed=0):
        super().__init__()
        random.seed(seed)
        self.hourly_arrival_rate = hourly_arrival_rate
        self.count = 0

        self.schedule([self.arrive])

    def arrive(self):
        print("{0}\tHello World! #{1}!".format(self.clock_time, self.count))
        self.count += 1
        self.schedule([self.arrive], datetime.timedelta(hours=round(random.expovariate(self.hourly_arrival_rate),2)))
```

The constructed child class HelloWorld takes in two inputs: the hourly arrival rate and the random seed. The latter being optional as it takes a default value of 0.

Under the initialization, the hourly arrival rate input is attached as a class attribute and a count attribute is constructed to take note the number of times an arrival event occur.

The schedule method of the sandbox model is called to put a pre-defined event into the future event list.

The arrival event is defined as a method of the class where it first prints "Hello World!" into the console, along with relevant information such as the arrival count and the time where the arrival executes. The method subsequently updates the count and schedule another arrival event at an interval that is exponentially distributed.

#### Running the Sandbox Model

``` python
if __name__ == '__main__':
    sim = HelloWorld(2, seed=10)
    sim.run(duration=datetime.timedelta(hours=5))
```

An instance of the sandbox model is first created with the required paramaters. The run method serves as a termination condition for the model.

In this example, the termination condition is defined as a time duration where the simulation is to run for 5 hours. After 5 hours, the model terminates, and any arrival events that happens after the 5th hour will not be of interest to the simulation.

#### Sample Output

``` python
0001-01-01 00:00:00	Hello World! #0!
0001-01-01 00:25:12	Hello World! #1!
0001-01-01 00:42:00	Hello World! #2!
0001-01-01 01:07:48	Hello World! #3!
0001-01-01 01:15:00	Hello World! #4!
0001-01-01 02:05:24	Hello World! #5!
0001-01-01 02:57:36	Hello World! #6!
0001-01-01 03:29:24	Hello World! #7!
0001-01-01 03:34:48	Hello World! #8!
0001-01-01 03:57:00	Hello World! #9!
0001-01-01 04:09:00	Hello World! #10!
0001-01-01 04:17:24	Hello World! #11!
```

## Cooperation and Contribution

Welcome to use O2DESpy. If you have any questions or suggestions, need help, fix bugs, contribute new modules, or improve the documentation, submit an issue in the community. We will reply to and communicate with you in a timely manner. 
