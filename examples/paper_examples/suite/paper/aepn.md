# **Action-Evolution Petri Nets: a Framework for Modeling and Solving Dynamic Task Assignment Problems** 

Riccardo Lo Bianco<sup>1</sup> , Remco Dijkman<sup>1</sup> , Wim Nuijten<sup>1</sup><sup>_,_2</sup> , and Willem van Jaarsveld<sup>1</sup> 

> 1 Eindhoven University of Technology, Netherlands 

_{_ `r.lo.bianco|r.m.dijkman|w.p.m.nuijten|w.l.v.jaarsveld` _}_ `@tue.nl` 

> 2 Eindhoven Artificial Intelligence Systems Institute, Netherlands 

**Abstract.** Dynamic task assignment involves assigning arriving tasks to a limited number of resources in order to minimize the overall cost of the assignments. To achieve optimal task assignment, it is necessary to model the assignment problem first. While there exist separate formalisms, specifically Markov Decision Processes and (Colored) Petri Nets, to model, execute, and solve different aspects of the problem, there is no integrated modeling technique. To address this gap, this paper proposes Action-Evolution Petri Nets (A-E PN) as a framework for modeling and solving dynamic task assignment problems. A-E PN provides a unified modeling technique that can represent all elements of dynamic task assignment problems. Moreover, A-E PN models are executable, which means they can be used to learn close-to-optimal assignment policies through Reinforcement Learning (RL) without additional modeling effort. To evaluate the framework, we define a taxonomy of archetypical assignment problems. We show for three cases that A-E PN can be used to learn close-to-optimal assignment policies. Our results suggest that A-E PN can be used to model and solve a broad range of dynamic task assignment problems. 

**Keywords:** Petri Nets, Dynamic Assignment Problem, Business Process Optimization, Markov Decision Processes, Reinforcement Learning 

## **1 Introduction** 

During the execution of a business process, tasks become executable and resources become available to execute these tasks. As resources are assigned to tasks, they become unavailable to execute other tasks. Consequently, continuously assigning the right task to the right resource is essential to run a process efficiently. This problem is known as dynamic task assignment. The dynamic task assignment problem can be seen as a particular case of the _dynamic assignment problem_ , which, according to [1], is the problem of assigning a fixed number of individuals to a sequence of tasks, such as to minimize the total cost of the allocations, which may include setup costs, travel costs, or other time-varying costs. 

2 Riccardo Lo Bianco et al. 

This problem has been extensively studied in business process optimization [2] as well as related areas, such as manufacturing [3]. For the sake of brevity, we will employ the term “assignment problem” to indicate the general dynamic (task) assignment problem. 

To solve an assignment problem, it must first be modeled mathematically. Markov Decision Processes (MDPs) are a common technique for modeling assignment problems [4], and they are the standard interface for Reinforcement Learning (RL) algorithms [5]. The basic definition of MDP involves a single agent interacting with an environment to maximize a cumulative reward, which is a global signal of the goodness of the actions chosen by the agent during a (possibly infinite) sequence of system states. In the context of business process optimization, the environment is the business process that must be executed, and the agent decides which task to assign to which resource. The reward is calculated based on what we want to optimize in the process, such as the total time resources spend working, the total cost of employing the resources, or the time customers spend waiting. While MDPs provide a good formalism for modeling the agent’s behavior, they consider the environment, in our case the business process, as a black box that provides rewards for the decisions taken by the agent without exposing its internal behavior. Moreover, they do not have an agreedupon syntax and lack any type of graphical representation. On the other hand, (Colored) Petri Nets [6] are a well-known formalism for modeling a business process but have no inherent mechanisms for modeling and calculating the best decision in a given situation. Also, frameworks exist for many mathematical optimization techniques, such as linear programming and constraint programming, where problems can be modeled and solved without additional effort. However, no such framework exists for dynamic task assignment problems. 

To fill this gap, this paper presents a unified and executable framework for modeling assignment problems. We use the term “unified” to refer to the capability of expressing both the agent and the environment of the assignment problem in a single standardized notation, thus simplifying the modeling of new problems. We use the term “executable” to refer to the possibility of using the models to train and test decision-making algorithms (specifically RL algorithms) without additional effort. To this end, we propose a new artifact in the form of a modeling language with a solid mathematical foundation, namely A-E Petri Net (A-E PN), which draws from the well-known Petri Net (PN) formalism to model assignment problems in a readable and executable manner. This paper pays particular attention to embedding the A-E PN formalism in the RL cycle, such that RL algorithms can be trained and used to solve assignment problems without additional effort. 

The proposed artifact is evaluated by modeling and solving a set of archetypical assignment problems. A taxonomy of assignment problem variants is proposed, and an example for each of the three main variants is modeled through A-E PN. An RL algorithm is trained on each instance, achieving close-to-optimal results. Apart from modeling each assignment problem as an A-E PN, no additional effort is required to achieve these results, empirically demonstrating that 

Action-Evolution Petri Nets 3 

A-E PN constitutes a unified and executable framework for modeling and solving assignment problems. 

Against this background, the remainder of this paper is structured as follows. Section 2 is dedicated to a review of relevant literature. Section 3 introduces Timed-Arc Colored Petri Nets (T-A CPN). Section 4 is devoted to the formal definition of Action-Evolution Petri Net and the description of the integration of A-E PN in the classic RL loop. In section 5, an essential taxonomy of assignment problem variants is presented. A problem instance for each variant is modeled through A-E PN, and a RL algorithm is trained on each instance, obtaining close-to-optimal results. Section 6 discusses the proposed method’s benefits and limitations and delineates the next research steps. 

## **2 Related work** 

To the best of our knowledge, this paper presents the first attempt at defining a unified and executable framework for assignment problems. In contrast, the relation between (generalized stochastic) Petri Nets and Markov Chains is well studied [7], but Markov Chains cannot be used to model and optimize (task assignment) decisions. Since Markov Decision Processes can be seen as an extension to Markov Chains, the idea of extending Petri Nets to model Markov Decision Processes follows naturally. Several attempts at this exist in the literature, but none focus on the assignment problem. An overview of existing frameworks for modeling and solving dynamic optimization problems is presented in Table 1, listing, for each framework, the Petri Net variant employed, the scope of applicability, and whether the framework is unified and executable. The current work is presented in the last line. 

**Table 1.** Comparison of existing frameworks for dynamic optimization. 

|Reference|PN|Scope|Unified|Executable|
|---|---|---|---|---|
|[8]|FPN|Problems expressible as finite MDPs|Yes|Yes*|
|[9]|DPN|Problems expressible as finite MDPs|No|Yes|
|[10]|GSPN|A single power management problem|Yes|No|
|[11]|TCPN|A single manufacturing scheduling problem|Yes|No|
|[12]|TCPN|Manufacturing scheduling problems|Yes|No|
|This paper|A-E PN|Assignment problems|Yes|Yes|



* No executable example is provided. 

In [8], the authors define a CPN variant: Factored Petri Net (FPN). In FPNs, the transition probabilities are defined explicitly, and a reward is attached to each network state. A limitation of [8] is that actions must be input marks from a single source transition (a transition without input arcs), while our framework allows actions to be defined anywhere in the Petri net, thus allowing for more modeling flexiblity. 

4 Riccardo Lo Bianco et al. 

In [9], the authors propose the Decision Petri Net (DPN) formalism. In DPN, the network is partitioned into a probabilistic network, in which transition probabilities are determined on arcs, and a non-deterministic network, corresponding to the actions that can be taken at a given moment by the decision maker. In our framework, we remove the need for two separate subnets and model the agents as tokens in the network, obtaining a unified representation. Both [8], and [9] require the number of states in the system to be finite, whereas our approach does not rely on states enumeration. 

In [10], the authors propose a model for a power-managed distributed computing system that is based on the Generalized Stochastic Petri Net (GSPN) formalism and provide a translation to the equivalent continuous-time MDP. The work demonstrates the expressive power of PN variants, but the resulting model is not executable. Also, the paper presents a single case study, while our approach is demonstrated to be generally applicable to modeling and solving problems with different characteristics. 

In [11], a manufacturing scheduling problem is modeled using Timed Colored Petri Nets (TCPN). The search for an optimal policy is implemented using Q- learning, where each action corresponds to a complete schedule, which is a path from the initial marking to a final marking of the TCPN representing the system, whereas in our case, an action corresponds to a single assignment, which allows for more flexible modeling of decisions. Moreover, [11] only covers a single case study, relying heavily on problem-specific heuristics. 

In [12], the authors provide an example usage of TCPN in the context of manufacturing systems, focusing on reinforcement learning as solving approach. While [12] highlights the relationship between TCPN and RL, TCPNs are used only to describe the environment and not to train or test solving algorithms. In contrast, our work provides a unified and executable framework. 

## **3 Preliminaries** 

This section provides the formal definition of Colored Petri Net (CPN) and Timed-Arc Colored Petri Net (T-A CPN), which will be used to define the new formalism. 

Colored Petri Net (CPN) [6] is an extension of Petri Nets (PN) in which tokens have different characteristics called colors. In the remainder of this section, we rely on the CPN definition provided in [13]. 

**Definition 1 (Colored Petri Net).** _A CPN is defined as a tuple CPN_ = ( _E, P, T, F, C, G, E, I_ ) _, such that:_ 

- _E is a finite set of types called color sets. Each color set must be finite and non-empty._ 

- _P is a finite set of places._ 

- _T is a finite set of transitions, such that P ∩ T_ = _∅_ 

- _F ⊆ P × T ∪ T × P is a finite set of arcs._ 

Action-Evolution Petri Nets 5 

- _C_ : _P →E is a color function that maps each place p into a set of possible token colors. Each token on p must have a color that belongs to the type C_ ( _p_ ) _, which is called the place’s color set._ 

- _G is a guard function. It is defined from T into expressions such that for each t ∈ T , G_ ( _t_ ) _is a Boolean expression and Type_ ( _V ar_ ( _G_ ( _t_ ))) _⊆E, where Type_ ( _x_ ) _denotes the type of x and V ar_ ( _f_ ) _denotes the set of free variables in the function f ._ 

- _E is an arc expression function. It is defined from F into expressions such that for each f ∈ F , Type_ ( _E_ ( _f_ )) = _C_ ( _P_ ( _f_ )) _MS and Type_ ( _V ar_ ( _E_ ( _f_ ))) _⊆E where P_ ( _f_ ) _is the place of f . This means that each evaluation of the arc expression must yield a multi-set (indicated by the MS subscript) over the color set attached to the corresponding place._ 

- _I is an initialization function. It is defined from P into expressions such that ∀p ∈ P_ : _Type_ ( _I_ ( _p_ )) = _C_ ( _p_ ) _MS. The initialization function determines the network’s initial marking._ 

**Definition 2 (Marking).** _A marking of a CPN is a function M , such that for each place p ∈ P , it defines a multi-set of colors C_ ( _p_ ) _→_ N _, which maps each possible color of the place to the number of times it occurs._ 

For a place _p_ with colors _C_ ( _p_ ) = _{c_ 1 _, c_ 2 _}_ , we also write _M_ ( _p_ ) = _c_ 1<sup>_ncm_</sup> 2<sup>to</sup> denote that _p_ has _n_ ) token with color _c_ 1 and _m_ tokens with color _c_ 2. Since a marking is a multi-set, multi-set operations, such as _≥_ , +, and _−_ , are available on markings. 

**Definition 3 (Binding).** _For a transition t, the variables V ar_ ( _t_ ) = _V ar_ ( _G_ ( _t_ )) _∪{V ar_ ( _E_ ( _f_ )) _|f ∈ F, T_ ( _f_ ) = _t} represent the set of variables from the guard function and the expressions on its arcs, where T_ ( _f_ ) _is the transition of arc f ._ 

_A binding of a transition t ∈ T is a function Y that maps each v ∈ V ar_ ( _t_ ) _to a color, such that ∀v ∈ V ar_ ( _v_ ) : _Y_ ( _v_ ) _∈ Type_ ( _v_ ) _and G_ ( _t_ ) _⟨Y ⟩ evaluates to true, where f ⟨Y ⟩ denotes the evaluation of a function f with its free variables bound as Y ._ 

For a transition _t_ with variables _V ar_ ( _t_ ) = _{v_ 1 _, v_ 2 _}_ , we also write _Y_ ( _t_ ) = _⟨v_ 1 = _c_ 1 _, v_ 2 = _c_ 2 _⟩_ to denote that the binding _Y_ assigns color _c_ 1 to variable _v_ 1 and color _c_ 2 to variable _v_ 2. 

We now define the behavior of a CPN through its firing rules. 

### **Definition 4 (CPN Firing Rules).** 

_1. A transition t is enabled in marking M for binding Y if and only if ∀_ ( _p, t_ ) _∈ F_ : _M_ ( _p_ ) _≥ E_ (( _p, t_ )) _⟨Y ⟩._ 

_2. An enabled transition can fire, changing the Marking M into a marking M_<sup>_′_</sup> _, such that ∀p ∈ P_ : _M_<sup>_′_</sup> ( _p_ ) = _M_ ( _p_ ) _− E_ (( _p, t_ )) _⟨Y ⟩_ + _E_ (( _t, p_ )) _⟨Y ⟩._ 

The standard CPN definition assumes that the effect of a firing is always instantaneous. To account for time, we will refer to a modified version of the 

6 Riccardo Lo Bianco et al. 

Timed-Arc Petri Net (T-A PN) formulation [14]. Our version defines a global clock, updated according to a next-event time progression. This is also the time management paradigm implemented in CPN Tools [15], a widely adopted software for Petri Nets modeling. 

**Definition 5 (Timed-Arc Colored Petri Net).** _A T-A CPN is defined by a tuple TACPN_ = ( _E, P, T, F, C, G, E, I_ ) _, where P, T, F, C, G, I are as in Definition 1, and E and E are adapted as follows:_ 

- _E is a finite set of timed types called timed color sets. A color of a timed color set has both a value v and a time τ , we also denote this as v_ @ _τ ._ 

- _E is an arc expression function. It is defined from F into tuples of two elements. For a given f ∈ F , E_ ( _f_ )0 _is defined the same as E in Definition 1 and E_ ( _f_ )1 _is a scalar increment, thus ∀f ∈ F_ : _Type_ ( _E_ ( _f_ )1) = N _, that indicates the generated tokens’ time with reference to the global clock. The second tuple element is ignored for arcs outgoing from places and incoming to transitions since the scalar increment is only used when producing new tokens._ 

Note that each color now has a time and consequently, each color in a marking and in a binding has time. For example, we can refer to the marking of a place _p_ with _M_ ( _p_ ) = _c_ 1@2<sup>1</sup> _c_ 1@3<sup>5</sup> as the marking that has one token with color _c_ 1 at time 2 and five tokens with color _c_ 1 at time 3. With some abuse of notation, we will allow arc expression functions _E_ ( _f_ )0, to ignore the time element of colors and leave it unaffected, and we will denote with _c_ @ _e_ that an expression _e_ only changes the time element of a timed color. 

We also extend the concept of marking to account for the presence of a global clock, which we need further on in the paper to define the transition rules for A-E PN. 

**Definition 6 (Timed Marking).** _A timed marking is defined as the tuple TM_ = ( _M, τ_ ) _, where M is a marking and τ is the current value of the global clock._ 

The T-A CPN firing rule can then be expressed as follows: 

### **Definition 7 (T-A CPN Firing Rules).** 

_1. Let t be a transition that is enabled in marking M for binding Y_ = _⟨v_ 1 = _c_ 1@ _τ_ 1 _, v_ 2 = _c_ 2@ _τ_ 2 _, . . . , vn_ = _cn_ @ _τn⟩ as in Definition 4 (using only E_ 0 _for E). The enabling time of the transition, denoted τE, is max_ ( _τ_ 1 _, τ_ 2 _, . . . , τn_ ) _._ 

_2. An enabled transition t is time-enabled in timed marking_ ( _M, τ_ ) _, if its enabling time τE is less than or equal to τ , and there exists no transition t_<sup>_′_</sup> _that is enabled in marking M for some binding Y_<sup>_′_</sup> _with enabling time τE_<sup>_′≤τE._</sup> 

_3. A transition t that is time-enabled in timed marking_ ( _M, τ_ ) _for binding Y with enabling time τE can fire, changing the timed marking to_ ( _M_<sup>_′_</sup> _, τE_ ) _, where M_<sup>_′_</sup> _is constructed, such that ∀p ∈ P_ : _M_<sup>_′_</sup> ( _p_ ) = _M_ ( _p_ ) _− E_ (( _p, t_ ))0 _⟨Y ⟩_ + _E_ (( _t, p_ ))0 _⟨Y ⟩_ @ _τE_ + _E_ (( _t, p_ ))1 _._ 

Action-Evolution Petri Nets 7 

_4. When there exists no t in timed marking_ ( _M, τ_ ) _, for which there is a binding Y , such that t is time-enabled, the global clock τ is increased until there is._ 

In practice, point 4 can be performed by evaluating bindings that are enabling but not time-enabling. The binding that leads to the lowest enabling time reveals the minimal increase of the global clock, making it possible to update the global clock using a next-event time progression. 

## **4 Action-Evolution Petri Nets** 

This section extends the definition of T-A CPN to provide a model that can automatically learn close-to-optimal task assignment policies. This extension is called Action-Evolution Petri Nets (A-E PN). The new elements are first described informally, then a formal definition is provided. Finally, the definition is incorporated into the RL cycle, allowing for automated learning of close-tooptimal task assignment policies. 

### **4.1 Tags and Rewards** 

The overall objective of A-E PN is to mimic the behavior of an agent that observes changes in the environment and acts upon those changes when possible. We will thus extend the CPN definition provided in the background section to distinguish two separate types of transitions: 

- **Actions** : transitions that represent actions taken by the agent. In the context of assignment problems, the firing of an action transition represents a single assignment. 

- **Evolutions** : transitions that represent events happening in the system independently of the actions taken by the agent. The firing of an evolution transition represents a single event in the environment, for example, the arrival of a new order. 

This distinction is expressed by associating every transition with a _transition tag_ , that can be either _A_ (action) or _E_ (evolution), through a _transition tag function L_ . We also extend the concept of marking to embed a _network tag l_ , which can assume a single value in _{A, E}_ : only transitions associated with a tag of the same type as the one in the network tag are allowed to fire. The network tag _l_ must be updated every time no transitions with the same tag are available for firing. The _tag update function S_ performs the update by changing the network’s tag from _A_ to _E_ or vice versa: _S_ ( _l_ ) = _A_ , if _l_ = _E_ ; _S_ ( _l_ ) = _E_ , if _l_ = _A_ . We use the term _tag time frame_ to refer to the period between changes in the network tag. 

The objective of the RL cycle is the maximization of a cumulative reward over a (possibly infinite) horizon. To track rewards in A-E PN, we introduce a _transitions reward function R_ that associates a reward to the firing of any transition, and we embed the total reward accumulated by firing transitions, which we call _network reward ρ_ , in the network’s marking. In general, a reward can be 

8 Riccardo Lo Bianco et al. 

produced by any change in the environment, regardless of whether an action or an evolution produced such change. For this reason, a reward is produced due to the firing of any transition, regardless if the transition is tagged as an action or an evolution. To comply with the classic RL cycle, rewards associated with evolutions are accumulated and awarded to the last action taken, eventually after a normalization operation (see subsection 4.3). 

To further clarify the basic mechanisms of A-E PN, the example in Fig. 1 provides an overview of a sequence of firings. 



<!-- Start of picture text -->
a b<br>Resources Resources<br>{a}@0 X {a}@0 X<br>{a}@0 {b}@0 {b}@0 X {a}@1 {b}@1 {a}@0{b}@0{b}@0 X<br>Waiting<br>X@+1 X@+1<br>E X X A X X E E X X A X@+1 X E<br>X X<br>Arrival Arrive Start Busy Complete Arrival Arrive Waiting Start Busy Complete<br>Guard Function:  None Guard Function:  None<br>Reward Function:  F(X) = 0 Reward Function:  F(X) = 1<br>d c<br>Resources Resources<br>{a}@1 X X<br>{b}@1 {b}@1 {b}@1<br>{a}@1 {a}@2 {b}@2 X {a}@1 X {a}@1 {b}@1<br>X@+1 X@+1<br>E X X A X@+1 X E E X X A X@+1 X E<br>X X<br>Arrival Arrive Waiting Start Busy Complete Arrival Arrive Waiting Start Busy Complete<br><!-- End of picture text -->

**Fig. 1.** A sequence of firings in a simple task assignment problem. 

The network shows the evolution of a system with two types of tasks, _a_ and _b_ , and two employees, one that can undertake only task _a_ and one that can undertake only task _b_ . A task of each type arrives at every clock tick, and an employee is assigned to a task of the same type. Assignments take one clock tick to complete, and a reward of 1 is produced every time an assignment is completed. The parentheses on the top right corner contain the components of the tagged marking that are not directly represented as network elements. Guard functions and reward functions are associated with single transitions. Timed tokens and arcs follow the notation introduced in Definition 5. The initial marking is presented in the dotted square _a_ , in which only _E_ transitions are enabled. After two firings of transition _Arrive_ , consuming both tokens in the _Arrival_ place (in any order), no evolution transitions are available, so the tag is updated, and the system transitions to state _b_ . Notice that the transition from _e_ to _a_ does not produce a clock update, since actions are available to be taken at time 0. In _b_ , transition _Start_ is enabled. In this case, the RL agent would have two available actions: pairing task _a_ with resource _a_ , or pairing task _b_ with resource _b_ . In this case, both actions will be taken sequentially, in any order, leading to tagged marking _c_ , while in the general case, choices would have to be made by a decision algorithm on which assignments to make. In _c_ , the 

Action-Evolution Petri Nets 9 

network tag is again _E_ , and two transitions are associated with time-enabled steps: _Arrive_ and _Complete_ . The firing of _Arrive_ produces two new tokens at time 1 in the _Waiting_ place, while the firing of _Complete_ places two tokens back in the _Resources_ place at time 1 and generates a network reward increment of 2 units in state _d_ . 

### **4.2 Formal Definition of Action-Evolution Petri Net** 

To provide a formal definition of A-E PN, we must adapt three definitions from T-A CPN: the net itself, the marking, and the firing rules. 

**Definition 8 (Action-Evolution Petri Net).** _Let T_ = _{A, E} be a finite set of tags representing actions and evolutions, and S_ : _T →T a network tag update function. An Action-Evolution Petri Net (A-E PN) is as a tuple AEPN_ = ( _E, P, T, F, C, G, E, I, L, lo, R, ρ_ 0) _, where E, P, T, F, C, G, E, I follow Definition 5, and:_ 

- _L_ : _T →T is a transition tag function that maps each transition t to a single tag. Only transitions associated with the same tag as the network can fire._ 

- _l_ 0 _∈T is a singleton containing the network’s initial tag, usually equal to E._ 

- **–** _R_ : _T →_ ( _f_ : R) _associates every transition with a reward function. The function can take timing properties or numbers of tokens (representing completed cases) as parameters, thus allowing for flexbility in modeling reward._ 

- **–** _ρ_ 0 _∈_ R _is the initial network reward, usually equal to_ 0 _._ 

= **Definition 9 (Tagged Marking).** _A tagged marking is a tuple TM_ ( _M, l, τ, ρ_ ) _, where the tuple_ ( _M, τ_ ) _is a timed marking, as in Definition 6, l ∈T is the network tag at the current time τ , and ρ ∈_ R _is the total reward accumulated until the current time τ ._ 

### **Definition 10 (A-E PN Firing Rule).** 

_1. A transition t is tag-enabled in a tagged marking_ ( _M, l, τ, ρ_ ) _for binding Y if and only if t is enabled in M according to Definition 1, and L_ ( _t_ ) = _l._ 

_2. Let t be a transition that is tag-enabled in tagged marking_ ( _M, l, τ, ρ_ ) _for binding Y_ = _⟨v_ 1 = _c_ 1@ _τ_ 1 _, v_ 2 = _c_ 2@ _τ_ 2 _, . . . , vn_ = _cn_ @ _τn⟩ . The enabling time of the transition, denoted τE, is max_ ( _τ_ 1 _, τ_ 2 _, . . . , τn_ ) _._ 

_3. An enabled transition t is tag-time-enabled in tagged marking TTM_ = ( _M, l, τ, ρ_ ) _, if its enabling time τE is less than or equal to τ , and there exists no transition t_<sup>_′_</sup> _that is enabled in tagged marking TTM for some binding Y_<sup>_′_</sup> _with enabling time τE_<sup>_′≤τE._</sup> 

_4. A transition t that is tag-time-enabled in tagged marking_ ( _M, l, τ, ρ_ ) _for binding Y with enabling time τE can fire, changing the tagged marking to_ ( _M_<sup>_′_</sup> _, l, τE, ρ_<sup>_′_</sup> ) _, where M_<sup>_′_</sup> _is constructed, such that ∀p ∈ P_ : _M_<sup>_′_</sup> ( _p_ ) = _M_ ( _p_ ) _− E_ (( _p, t_ ))0 _⟨Y ⟩_ + _E_ (( _t, p_ ))0 _⟨Y ⟩_ @ _τE_ + _E_ (( _t, p_ ))1 _and ρ_<sup>_′_</sup> = _ρ_ + _R_ ( _t_ ) _._ 

10 Riccardo Lo Bianco et al. 

_5. When there exists no t in tagged marking TTM_ = ( _M, l, τ, ρ_ ) _, for which there is a binding Y , such that t is time-enabled, the set of all transitions is partitioned in two disjoint sets: Tcurrent_ = _{t ∈ T |L_ ( _t_ ) = _l} and Tnext_ = _{t ∈ T |L_ ( _t_ ) _̸_ = _l}. Let τcurrent be the minimum value for which a transition in Tcurrent is time-enabled (according to Definition 7), and let τnext be the minimum value for which a transition in Tnext is time-enabled. Note that τcurrent and τnext can be undefined._ 

   - _If τcurrent is defined, and τcurrent ≤ τnext or τnext is undefined, only the global clock is updated, leading to a new tagged marking TTM_<sup>_′_</sup> = ( _M, l, τcurrent, ρ_ ) _._ 

   - _If τnext is defined, and τcurrent > τnext or τcurrent is undefined, both the global clock and the network tag are updated, leading to a new tagged marking TTM_<sup>_′_</sup> = ( _M, S_ ( _l_ ) _, τnext, ρ_ ) _._ 

### **4.3 Extending the Reinforcement Learning Loop** 

Having completely defined the characteristics of the A-P PN formalism, we can clarify how it can be used to learn optimal task assignment policies (i.e. mapping from observations to assignments) by applying it in a Reinforcement Learning (RL) cycle. Figure 2 shows the RL cycle. In every step in the cycle, the agent receives an observation (a representation of the environment’s state), then it produces a single action that it considers the best action for this observation. The action leads to a change in the environment’s state. The environment is responsible for providing a reward for the chosen action along with a new observation. Then the cycle repeats, and a new decision step takes place. The MDP formulation is the standard framework for training an agent to take actions that lead to the highest cumulative reward. 



<!-- Start of picture text -->
observation  Agent<br>action<br>reward<br>Environment<br><!-- End of picture text -->

**Fig. 2.** A common representation of the RL training cycle [5]. 

In recent years, the embedding of neural networks in RL algorithms gave birth to the field of Deep Reinforcement Learning (DRL), achieving breakthroughs in settings such as playing board games [16] and robotic manipulation [17], as well as successful applications in domains like industrial process control [18], and healthcare [19]. With the proliferation of robust DRL algorithms, the main hurdle in modeling new problems is the definition of the environment, which is usually represented as a black box, as in Fig. 2, thus leaving the implementation 

Action-Evolution Petri Nets 11 

of the system’s dynamics entirely to the modeler. The lack of a standardized interface makes the creation of new environments time-consuming and dependent on the modeler’s coding skills. Moreover, even introducing small changes potentially requires substantial effort once the environment has been modeled. These observations motivate the effort to provide a unified and executable framework. In Fig. 3, the classic RL cycle is extended to account for the presence of A-E PN. The main element is the A-E PN, which acts as a simulator for the whole process. 



<!-- Start of picture text -->
observation  Agent<br>action<br>reward<br>Environment<br>A<br>Network Observation Action<br>Tag? Manager Manager<br>E marking  binding<br>network reward<br>A-E PN<br><!-- End of picture text -->

**Fig. 3.** The reinforcement learning cycle with A-E PN 

The A-E PN communicates with the agent through two sub-components: _observation manager_ and _action manager_ . The observation manager is invoked every time the tagged marking changes, regardless if due to a firing or not. The new reward is stored, and the network tag is evaluated: if the tag is _E_ , no action is required, and the control is given back to the A-E PN, which can fire a new _E_ transition. If the tag is _A_ , the accumulated rewards are added up, and the result is divided by 1 + ( _τt_ +1 _− τt_ ). The resulting value is returned to the agent as _rt_ +1. The reward value takes into account the possible misalignment between clock ticks ( _τ_ ) and RL steps ( _t_ ), given by the fact that multiple actions can happen at the same _τ_ . The observation manager also returns to the agent the new observation _ot_ +1. For the set of experiments presented in the next section, the observation is built as a vector containing, for each place, the number of tokens of each color in the place’s color set. The action manager is invoked every time the agent chooses an action _at_ , which it transforms into the corresponding binding _Bt_ (associated with an action transition) to be fired. 

## **5 Evaluation** 

This section aims to show that A-E PN constitutes a unified and executable framework for expressing dynamic task assignment problems with different characteristics: in fact, all the examples were modeled using a single notation (except 

#### 12 Riccardo Lo Bianco et al. 

for color-specific functions on arcs, guards, and rewards) and a RL algorithm was trained on each problem, without any additional development effort. 

We provide a (non-exhaustive) taxonomy of assignment problem variants based on [20]. We distinguish three archetypes of assignment problems. 

- **Assignment Problem with Compatibilities** : resources are assigned to tasks according to a measure of compatibility. Two problem subclasses can be formulated: 

   - **Assignment Problem with Hard Compatibilities** : resources can only be assigned to tasks if they are compatible. The dynamic task assignment problem in subsection 5.1 falls into this subclass. 

   - **Assignment Problem with Soft Compatibilities** : resources can always be assigned to tasks, but different assignments result in different system behaviors. An example of such a problem is if multiple resources can perform a task, but some will be faster at it than others. 

- **Assignment Problem with Multiple Assignments** : the same resource can be assigned to multiple tasks, or the same task can be assigned to multiple resources. Two problem subclasses can be formulated: 

   - **Assignment Problem with Resource Capacity** : resources have a maximum capacity of tasks that they can undertake before being considered full. In the simple case each resource can only be busy with a single task at a time. The dynamic bin packing problem in subsection 5.2 provides a more elaborate example. 

   - **Assignment Problem with Task Capacity** : tasks have a minimum capacity of resources to be assigned to them before processing. In the simple case each tasks needs exactly one resource. 

- **Assignment Problem with Dynamic Resources’ Behavior** : resources have dynamic behavior. Two problem subclasses can be formulated: 

   - **Assignment Problem with Action-Dependent Dynamic Resources’ Behavior** : resources change their attribute values as the consequence of taking actions. The dynamic order-picking problem in subsection 5.3 falls into this category. 

   - **Assignment Problem with Action-Independent Dynamic Resources’ Behavior** : resources change their attribute values as the consequence of evolutions in the environment. For example, resources may take breaks or go on holidays. 

In the following sections, one example is detailed for each archetype. An example for each subclass is implemented in the provided Python package. 

### **5.1 Dynamic Task Assignment Problem with Hard Compatibilities** 

Let us consider a system that solves a task assignment problem, similar to the one presented in Fig. 1. At every clock tick, two tasks arrive: one has type _r_ 1 and the other _r_ 2. Two resources are available for the assignment: one can only undertake tasks of type _r_ 1, while the other can undertake tasks of type _r_ 1 or _r_ 2. Once a 

Action-Evolution Petri Nets 13 

task is assigned to a resource, completion always takes one clock tick, after which the resource becomes available for a new assignment. A resource cannot work on multiple tasks at the same time. A network reward of 1 is returned every time a task is assigned to a resource and every time an assignment completes, leading to a theoretical maximum reward of 200 over 100 clock ticks. The problem can be fully expressed in terms of A-E PN, as reported in Fig. 4. 



<!-- Start of picture text -->
Resources<br>{r1;r2}@0 Y<br>{r2}@0<br>{r1}@0<br>Y<br>{r1}@0<br>X@+1<br>E X X A (X;Y)@+1 (X;Y) E<br>X<br>Arrival Arrive Waiting Start Busy Complete<br>Guard Function:  None Guard Function:  compatible(X, Y) Guard Function:  None<br>Reward Function:  F(X) = 0 Reward Function:  F(X) = 0 Reward Function:  F(X) = 1<br><!-- End of picture text -->

**Fig. 4.** A-E PN initial marking for the dynamic task assignment problem 

### **5.2 Dynamic Bin Packing Problem** 

In this scenario, we model a dynamic version of the bin packing problem where items (the problem tasks, characterized by their _weight_ ) arrive sequentially and they must be allocated to two bins (the problem resources, characterized by the total weight of objects in the bin _curr_ and the bin’s total capacity _tot_ ) that are emptied at every clock tick (except for the first, which is used to generate the objects to be put in the bins). The fullness of the bins before being emptied gives the measure of goodness of the object’s allocation, quantified as the weight of objects in the bin divided by the total bin capacity. This problem showcases how tokens’ colors can be used to model non-trivial reward functions. In the example reported, three objects arrive in the system at every clock tick, one of weight 1 and two of weight 2. Two initially empty bins are available, one with capacity 2 and one with capacity 3. The optimal allocation would give a reward of 2, leading to a theoretical maximum reward of 200 over a 100 clock ticks horizon. The A-E CPN formalization of the problem is reported in Fig. 5. 

### **5.3 Dynamic Order-Picking Problem** 

In this section, we present an example of action-dependent resource behavior (i.e. the agent taking decisions on the actions that it performs). The example is a simple order-picking problem in which a single agent (the resource) moves on 

14 Riccardo Lo Bianco et al. 



<!-- Start of picture text -->
{weight: 2}@0 {weight: 1}@0 {curr: 0, tot: 3}@1<br>{curr: 0, tot: 2}@1<br>X@+1 Y {curr: 0, tot: X.tot}@+1<br>E X@+1 X A E<br>X {curr: Y.curr+X.weight, tot: Y.tot} X<br>Arrival Arrive Ready Assign Bins Empty<br>Guard Function:  None Guard Function:  Y.tot-Y.curr >= X.weight Guard Function:  None<br>Reward Function:  F(X) = 0 Reward Function:  F(X) = 0 Reward Function:  F(X) = X.curr/X.tot<br><!-- End of picture text -->

**Fig. 5.** A-E PN initial marking for the dynamic bin packing problem 

a squared grid of size 2, trying to pick orders (the tasks). The agent’s and the orders’ colors are characterized by two parameters representing the coordinates on the grid (infinite capacity is assumed). The agent starts in position (0 _,_ 0) and can move left, right, up, or down, but not over a diagonal. If an order is in the same position as the agent, the latter can use an action to pick the order. A single order arrives at every clock tick, always in position 1 _,_ 1, and the order stays on the grid for exactly one clock tick, according to a time-to-live (TTL) parameter. The agent’s objective is to pick as many orders as possible, so it gets a reward of 1 every time an order is picked, leading to a theoretical maximum reward of 98 over a 100 clock ticks horizon (at least two orders will be lost due to the agent moving to position (1 _,_ 1). The problem is formulated in terms of A-E PN in Fig. 6. 



<!-- Start of picture text -->
Guard Function:  adjacent(X, Y) {x: 0, y: 0}@0<br>Reward Function:  F(X) = 0<br>{x: 0, y: 0}@0 Pickup Guard Function:  X.x == Y.x and X.y == Y.y<br>{x: 0, y: 1}@0 Y Y@+1 X<br>{x: 1, y: 0}@0 A A Reward Function:  F(X) = 1<br>{x: 1, y: 1}@0 Y X X<br>Positions Move Guard Function:  X.ttl == 0<br>{x: 0, y: 0, ttl: 1}@0 Y Lose Reward Function:  F(X) = 0<br>Ready<br>X@+1<br>E X@+1 X E<br>X<br>Arrival Arrive<br>X {x: X.x, y: X.y, ttl: X.ttl-1}<br>Guard Function:  None E<br>Reward Function:  F(X) = 0<br>Decrement_ttl<br><!-- End of picture text -->

**Fig. 6.** A-E PN initial marking for the dynamic order-picking problem 

Action-Evolution Petri Nets 

15 

### **5.4 Experimental Results** 

All experiments were implemented in a proof-of-concept package<sup>1</sup> , relying on the Python programming language and the widely adopted RL library Gymnasium [21]. Proximal Policy Optimization (PPO) [22] with masking was used as the training algorithm. Specifically, the PPO implementation of the _Stable Baselines_ package [23] is used. Note, however, that the mapping from each A-E PN to PPO was automated and requires no further effort from the modeler. The PPO algorithm was trained on each example for (10<sup>6</sup> steps with 100 clock ticks per episode, completed in less than 2300 seconds on a mid-range laptop, without GPUs), always using the default hyperparameters. The experimental results were computed on (network) rewards obtained by the trained agent and following a random policy over 1000 trajectories, each of duration 100 clock ticks. In Table 2, the average and standard deviations of rewards obtained by the trained PPO are compared to those of a random policy on each of the three presented problem instances, with reference to the maximum attainable reward. In all cases, PPO shows to be able to learn a close-to-optimal assignment policy. 

**Table 2.** The results for the three presented problem instances. 

|Instance|Random|PPO|Optimal|
|---|---|---|---|
|Task Assignment|186_._894_±_2_._084|199_._852_±_0_._398|200|
|Bin Packing|186_._746_±_1_._941|199_._963_±_0_._186|200|
|Order Picking|6_._046_±_2_._585|96_._776_±_2_._019|98|



## **6 Conclusions and Future Work** 

This paper presented a framework for modeling and solving dynamic task assignment problems. To this end, it introduced a new variant of Petri Nets, namely Action-Evolution Petri Nets (A-E PN), to provide a mathematically sound modeling tool. This formalism was integrated with the Reinforcement Learning (RL) cycle and consequently with existing algorithms that can solve RL problems. To evaluate the general applicability of the framework for modeling and solving task assignment problems, a taxonomy of archetypical problems was introduced, and working examples were provided. A DRL algorithm was trained on each implementation, obtaining close-to-optimal policies for each example. This result shows the suitability of A-E PN as a unified and executable framework for modeling and solving assignment problems. 

While the applicability of the framework was shown, its possibilities and limitations are yet to be fully explored. This will be done in future research by expanding the provided taxonomy of assignment problems and considering different problem classes. 

> 1 The code is publicly available in https://github.com/bpogroup/aepn-project. 

16 Riccardo Lo Bianco et al. 

## **Acknowledgement** 

The research that led to this publication was partly funded by the European Supply Chain Forum (ESCF) and the Eindhoven Artificial Intelligence Systems Institute (EAISI) under the AI Planners of the Future program. 

## **References** 

1. H. W. Kuhn. The Hungarian method for the assignment problem. _Naval Research Logistics Quarterly_ , 2(1-2):83–97, March 1955. 

2. N. G¨ulpınar, E. C¸anako˘glu, and J. Branke. Heuristics for the stochastic dynamic task-resource allocation problem with retry opportunities. _European Journal of Operational Research_ , 266(1):291–303, April 2018. 

3. L. Hu, Z. Liu, W. Hu, Y. Wang, and J. Tan. Petri-net-based dynamic scheduling of flexible manufacturing system via deep reinforcement learning with graph convolutional network. _Journal of Manufacturing Systems_ , 55:1–14, April 2020. 

4. M. Z. Spivey and W. B. Powell. The Dynamic Assignment Problem. _Transportation Science_ , 38(4):399–419, November 2004. 

5. R. Sutton and A. Barto. Reinforcement Learning: An Introduction. page 352. 

6. K. Jensen. A brief introduction to coloured Petri Nets. In _Tools and Algorithms for the Construction and Analysis of Systems_ , volume 1217, pages 203–208. 1997. 

7. F. Bause and P. Kritzinger. Stochastic Petri Nets: An Introduction to the Theory. _ACM SIGMETRICS Performance Evaluation Review_ , 26(2):2–3, August 1998. 

8. M. Eboli and F. Cozman. Markov Decision Processes from Colored Petri Nets. In _Advances in Artificial Intelligence – SBIA 2010_ , volume 6404, pages 72–81. 2010. Series Title: Lecture Notes in Computer Science. 

9. M. Beccuti, G. Franceschinis, and S. Haddad. Markov Decision Petri Net and Markov Decision Well-Formed Net Formalisms. In _Petri Nets and Other Models of Concurrency – ICATPN 2007_ , volume 4546, pages 43–62. 2007. 

10. Q. Qiu, Q. Wu, and M. Pedram. Dynamic power management of complex systems using generalized stochastic Petri nets. In _Proceedings of the 37th conference on Design automation - DAC ’00_ , pages 352–356. ACM Press, 2000. 

11. M. Drakaki and P. Tzionas. Manufacturing Scheduling Using Colored Petri Nets and Reinforcement Learning. _Applied Sciences_ , 7(2):136, February 2017. 

12. S. Riedmann, J. Harb, and S. Hoher. Timed Coloured Petri Net Simulation Model for Reinforcement Learning in the Context of Production Systems. In Bernd-Arno Behrens, Alexander Brosius, Welf-Guntram Drossel, Wolfgang Hintze, Steffen Ihlenfeldt, and Peter Nyhuis, editors, _Production at the Leading Edge of Technology_ , pages 457–465, Cham, 2022. Springer International Publishing. 

13. K. Jensen and G. Rozenberg. _High-level Petri nets: theory and application_ . Springer-Verlag, 1991. 

14. L. Jacobsen, M. Jacobsen, M. H. Møller, and J. Srba. Verification of timed-arc petri nets. In _SOFSEM 2011: Theory and Practice of Computer Science_ , pages 46–72, 2011. 

15. CPN Tools. https://cpntools.org/. 

16. D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis. Mastering the game of Go without human knowledge. _Nature_ , 550(7676):354–359, October 2017. 

Action-Evolution Petri Nets 17 

17. D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke, and S. Levine. QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation, November 2018. 

18. R. Nian, J. Liu, and B. Huang. A review on reinforcement learning: Introduction and applications in industrial process control. _Computers & Chemical Engineering_ , 139:106886, August 2020. 

19. C. Yu, J. Liu, and S. Nemati. Reinforcement Learning in Healthcare: A Survey, April 2020. arXiv:1908.08796 [cs]. 

20. David W. Pentico. Assignment problems: A golden anniversary survey. _European Journal of Operational Research_ , 176(2):774–793, January 2007. 

21. Gymnasium. https://gymnasium.farama.org/. 

22. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy Optimization Algorithms, August 2017. arXiv:1707.06347 [cs]. 

23. Sb3-contr. https://github.com/Stable-Baselines-Team/stable-baselines3-contrib. 

