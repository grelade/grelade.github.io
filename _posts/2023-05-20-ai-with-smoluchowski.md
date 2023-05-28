---
title: "AI a la Smoluchowski"
layout: post
usemathjax: true
---

![frontiou](/assets/posts/2023-05-20/front.png)

Heard of Smoluchowski? Interested in AI? Stumbled lately on these breathtaking AI-generated images? Why am I asking you these questions? Find out here.



---

## So there's this guy...

So there's this forgotten polish physicist, [Marian Smoluchowski](https://en.wikipedia.org/wiki/Marian_Smoluchowski). In his short life he became one of the pioneers of statistical physics. One of his **major achievements** was to **explain** the **erratic motion of pollen grain** in water first reported by Brown: 

{% figure caption:"extracted from a [clip](https://www.youtube.com/watch?v=R5t-oA796to)" %}
![](/assets/posts/2023-05-20/brownian.gif){:width="35%"}
{% endfigure %}

Why is it so strange? It seems that the **grains move** quite **randomly** due to interaction with the much smaller fluid particles... But hold on. Some context is urgently needed - as shocking as it sounds today, in the **mid XIXth century**, the **existence of atoms** themselves was **a highly controversial idea**! Some argue that [Ludwig Boltzmann](https://en.wikipedia.org/wiki/Ludwig_Boltzmann), the grandfather of statistical physics, took his own life because of brutal attacks against his microscopic, atomistic description of matter. **If** however there are **no atoms**, the **fluid** is a **continuous, featureless medium** and there is no reason for the erratic motion of pollen grains to happen. Still, as was shown by Smoluchowski, the motion is the net result of many interactions with small, invisible fluid molecules:

{% figure caption:"simulation [code](https://github.com/Yangliu20/physics-simulation)" %}
![](/assets/posts/2023-05-20/brownian_sim.gif){:width="25%"}
{% endfigure %}

While he was not the only one to explain the motion (Einstein was also on it!), he embraced the atomistic perspective fully. In his seminal work[^1], he derived a fundamental result for the **average particle position** changing as a **square root of time**:

{% figure %}
![](/assets/posts/2023-05-20/basic-eq.png){:width="20%"}
{% endfigure %}

This formula set out a number of further developments. The trembling motion is described fully by **Smoluchowski equation**[^2]:

$$
 \frac{\partial}{\partial t} p(x,t) = D \frac{\partial^2}{\partial x^2} p(x,t)
$$

where we restrict the motion to one dimension $$x$$, $$D$$ is a constant. Equation specifies $$p(x,t)$$, the probability of finding a particle at $$x$$ in time $$t$$. The motion governed by this equation is known as the **free diffusion** since nothing is constraining the random motion.

#### Simulations

That was just the beginning of the story. Later, **Langevin** realized that **single particles** undergoing Brownian motion can be **described in terms of an ODE** by introducing a random (noise) function $$W(t)$$:

$$
\frac{d}{dt}x(t) = -ax(t) + \sqrt{2D} W(t)
$$

This diffusive motion is constrained by a restoring force $$-ax$$ keeping the particle from moving away to infinity (setting $$a=0$$ takes us back to the free diffusion discussed previously). **Langevin** formulation is very **easy to simulate**, we approximate the derivative $$dx/dt \approx ( x(t+\delta t) - x(t) )/ \delta t$$ and set time $$t = n \delta t$$ while $$x(n\delta t) = x_n$$ and $$W(n\delta t) \sqrt{\delta t} = W_n $$:

$$
x_{n+1} = (1- a \delta t ) x_n + \sqrt{2D} W_n \sqrt{\delta t}
$$

The nontrivial part is a little bit of random function magic[^3] producing the square root $$\sqrt{\delta t}$$. We simulate it for some initial position $$x_0$$ and for $$N$$ time-steps. It is instructive to **compare single trajectories** produced by the Langevin approach matched **with the probabilistic description** provided by the Smoluchowski equation:

{% figure caption:"Comparison between (single-particle) Langevin equation and (probabilistic) Smoluchowski equation" %}
![](/assets/posts/2023-05-20/simple-diff.png){:width="80%"}
{% endfigure %}

In simulations we set $$a=0$$ resulting in free,unconstrained diffusive dynamics. Simulated trajectories are shown in the left plot up to time $$T = 2$$. The right plot evaluates both approaches at fixed time $$t=1$$, denoted by a red vertical line. The histogram is a result of binning the trajectories while the black line is an explcit solution to the Smoluchowski equation $$p(x,t) \sim \exp \left ( - x^2 / 4D t \right )$$.

---

## ... But where is AI?

OK, it's a nice story and all but... I came here because of the AI, **what does it have to do with the cool stuff** people do nowadays? Well, let's look at the paper starting the newest generative craze **"Denoising Diffusion Probabilistic Models"** or the name [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release). They happen to create truly breathtaking results:

{% figure caption:"*self*-created using [Midjourney](https://www.midjourney.com/)" %}
![](/assets/posts/2023-05-20/midjourney-example.png){:width="50%"}
{% endfigure %}

How is this possible? Simply put - it is precisely **diffusion generalized to multiple dimensions** which turns out to be **quite powerful**. There are two main parts of a generative diffusion model.

#### Forward process

We first define the so-called **forward process** -- we take an initial image and gradually add noise to each pixel. In that way, we create a set of images with increasing noise-level. Not focusing on minor details[^4], the end result is just pure noise. 

#### Reverse process

The key phase is to consider the **reverse process**. We **start from complete noise** and do a backward simulation to **generate an image**... But wait a second, how can this happen? After all, noise is featureless, lacks any initial information about the images. That's true and we need some ML magic here - we **need a powerful constraining function** to **drive the initial noise** into something resembling images. A trainable **neural network** will fulfill this role quite well. 

Let's now look how we do that. Formally, we can write down the **Langevin** equation for the **reverse process**. Starting from a noisy $$x(T)$$, it is described by 

$$
\frac{d}{dt} x(t) = -ax(t) + s_\theta(x(t),t) + \sqrt{2D} W(t), \qquad t \in (0,T)
$$

with the crucial driving function $$s_\theta$$ (aka the **score function**) depending on parameters $$\theta$$. 

#### Learning the score function $$s_\theta$$ 

This phase is typical in ML applications. Once we know the optimization criteria for the model, we are ready to go. In our case, the **learning minimizes** the following **loss** function[^5]

$$
L(\theta) = E_{t\sim U(0,T)} E_{x(0)} E_{x(t)| x(0)} \left ( \lambda(t) \left \| s_\theta(x(t),t) - \partial_{x(t)} \log p(x = x(t), t; x=x(0),t=0) \right \|^2 \right )
$$

where the **expectation values** are taken over
- time $$t$$ (uniformly)
- initial images $$x(0)$$ (samples)
- noisy images $$x(t)$$ at time $$t$$ resulting from $$x(0)$$ (generated by the forward process)

The **transition kernel** $$p(x, t; x_0,0)$$ is a **fundamental solution** to the **Smoluchowski** equation while $$\lambda(t)$$ is the time-weighting function. 

#### Image generation

Output of the training phase is the **optimal score function $$s_{\theta_*}$$**. It is used in the generative reverse process. Because it evolves backward in time, we use the retarded approximation to time derivative $$dx/dt \approx ( x(t) - x(t-\delta t))/ \delta t$$ and the discretized Langevin equation reads

$$
x_{n-1} = (1 + a \delta t) x_n - s_{\theta_*}(x_n,n\delta t) \delta t + \sqrt{2D} W_n \sqrt{\delta t}, \quad n = N,N-1,...,1 .
$$

where the initial value $$x_N = x(T)$$ is drawn from the normal distribution. If the learning process was succesful, the end result is a sample $$x_0$$ drawn from the learned image distribution.

---

## Toy example or positions vs images

By now, **diffusion models** became indispensable in the **state-of-the-art image generation** with many good resources available on the topic. To end this post I **discuss the simplest** diffusion model consisting of a **shallow feed-forward neural network** that generates fairly well samples from a **2D synthetic dataset**. For those interested in the code, check out the [NOTEBOOK](https://github.com/grelade/nano-diffusion/blob/master/notebook.ipynb).

#### Specyfing the dataset

First we turn to **specifying the dataset**. We implement the model with ``pytorch``, the function to generate the samples:

```python
def gen_x0(n = 2,scale = 200):
    x0 = 2*torch.rand(n)-1
    x0 /= torch.sqrt(((x0)**2).sum())
    x0 *= torch.distributions.gamma.Gamma(concentration=scale,rate=scale).sample()
    return x0
```

It is concentrated on a ring of radius 1 with nonuniform angular density. The ring thickness is controlled by the  ``scale`` parameter. We show a 2D histogram of the dataset, 1D radial and 1D angular histograms:

{% figure %}
![](/assets/posts/2023-05-20/toy-dataset.png){:width="90%"}
{% endfigure %}

#### Forward process

The Langevin equation for the forward process is the continuous version considered in the seminal paper[^6]:

$$
\frac{d}{dt} x(t) = - \frac{1}{2} \beta(t) x(t) + \sqrt{\beta(t)} W(t), \qquad t \in (0,T)
$$

where the noise schedule is linear $$\beta(t) = \beta_- + (\beta_+ - \beta_-) \frac{t}{T}$$. The solution to an initial value problem is explicitly given by $$ x(t) = \sqrt{c(t)}x(0) + (1-c(t)) \epsilon$$ with $$c(t) = \exp (-\int_0^t \beta(t') dt')$$ and random $$\epsilon \sim N(0,1)$$ drawn from a normal distribution. The function for generating a forward process reads:

```python
def forward_x(x0,t,noise = None, beta_min = 1e-4, beta_max = 0.02, T = 100):
    c = np.exp(-beta_min*t - 1/(2*T)*(beta_max-beta_min)*t**2)
    eps = torch.randn_like(x0) if noise is None else noise
    return np.sqrt(c)*x0 + (1-c)*eps
```

The ``noise`` tensor can be passed in order to use the same noise sample.

#### Reverse process

The reverse process[^5] matched to the forward process is given by the Langevin equation:

$$
\frac{d}{dt} x(t) = - \frac{1}{2} \beta(t) x(t) - \beta(t) s_\theta(x(t),t) + \sqrt{\beta(t)} W(t), \qquad t \in (0,T)
$$

In this case, we start out from $$x(T) \sim N(0,1)$$ drawn from a normal distribution and evolve to $$x(0)$$. The function for the backward process reads

```python
def reverse_xT(xT,score_net, beta_min = 1e-4, beta_max = 0.02, T = 100, dt = 1.0):
    
    def beta(t):
        return beta_min + (beta_max-beta_min)*t/T

    def reverse_step(xt,t):
        tmat = (torch.ones_like(xt)*t)[...,:-1]
        xt_new = xt + (1/2*beta(t)*xt+ beta(t)*score_net(xt,tmat))*dt 
        xt_new += np.sqrt(beta(t))*torch.randn_like(xt)*np.sqrt(dt)
        return xt_new

    xts = xT.unsqueeze(-2)
    
    N = int(T/dt)
    for n in range(N,0,-1):
        xt = reverse_step(xts[...,-1:,:],n*dt)
        xts = torch.cat((xts,xt),axis=-2)
        
    return xts
```

#### Neural network 

The ``score_net`` passed to the ``reverse_xT`` function is the discussed neural network. To ease the discussion, we aim at the **simplest network that does the job**. So there will be **no UNets** here[^7] but a feed-forward network with four layers. The **position** ``x`` and **time** ``t`` parameters are **simply concatenated** before passing through the network:

```python
class basic_net(nn.Module):
    def __init__(self, dim = 2, layer_size = 64, n_layers = 4):
        
        super().__init__()
        self.activ = nn.SiLU()

        self.layers = []
        self.layers.append(nn.Linear(dim+1,layer_size))
        for _ in range(n_layers-2):
            self.layers.append(nn.Linear(layer_size,layer_size))
        self.layers.append(nn.Linear(layer_size,dim))
        self.layers = nn.ModuleList(self.layers)

    def forward(self,x,t):

        xt = torch.cat((x,t),axis=-1)        
        for i, l in enumerate(self.layers[:-1]):
            xt = self.activ(l(xt))
        xt = self.layers[-1](xt)
    
        return xt
```

#### Learning phase

With an explicit solution to the forward process, the transition kernel $$p(x, t; x=x_0,t=0) \sim \exp \left ( - \frac{\left (x-x_0 \sqrt{c(t)} \right )^2}{2(1-c(t))^2} \right )$$ so that the score function needed in the learning phase reads:

$$
\partial_x \log p(x, t; x=x_0,t=0) =  - \frac{\left (x-x_0 \sqrt{c(t)} \right )}{(1-c(t))^2}
$$

To **simplify the learning**, we plug to the score function an explicit form for the noisy example $$x(t) = \sqrt{c(t)}x(0) + (1-c(t)) \epsilon$$ resulting in $$\partial_{x(t)} \log p(x = x(t), t; x=x(0),t=0) = - \frac{\epsilon}{1-c(t)}$$. The loss function reads

$$
L(\theta) = E_{t\sim U(0,T)} E_{x(0)} E_{\epsilon} \left ( \lambda(t) \left \| s_\theta\left (\sqrt{c(t)}x(0) + (1-c(t)) \epsilon,t \right ) + \frac{\epsilon}{1-c(t)}\right \|^2 \right )
$$

Optimizing **current loss** function **will be unstable** due to the $$1-c$$ term in the denominator which vanishes for $$t\approx 0$$. To **help with** these **convergence** issues, we pick a weight function $$\lambda(t) = \sqrt{1-c(t)}$$. Full training loop is given below:

```python
def c(t, beta_min = 1e-4, beta_max = 0.02, T = 100):
    return np.exp(-beta_min*t - 1/(2*T)*(beta_max-beta_min)*t**2)

batch_size = 512
n_epochs = 80
n_batches = 200
T = 100

score_net = basic_net()
optimizer = torch.optim.AdamW(score_net.parameters(),lr=0.01)
loss_f = nn.MSELoss()

for n in range(n_epochs):
    for i in range(n_batches):
        
        x0 = torch.stack([gen_x0() for _ in range(batch_size)])
        t = T*torch.rand(batch_size,1)

        noise = torch.randn_like(x0)
        xt = forward_x(x0,t,noise = noise)

        score_0 = -noise
        score_1 = (1-c(t))*score_net(xt,t)
        loss = loss_f(score_0,score_1)
        
        loss.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    prnt(f'{n+1}/{n_epochs}: loss = {loss.detach().item()}')

```

The result after $$80$$ epochs **looks quite good**. All angular **peaks are recreated** although the four-fold symmetry was not captured correctly. The **radial distribution is recreated** very well.

{% figure caption:"Toy dataset sampled from a learned diffusion model." %}
![](/assets/posts/2023-05-20/learned-dataset.png){:width="90%"}
{% endfigure %}

#### Conclusions and improvements

* Our toy diffusion model based on a **four-layer** feed-forward neural **network learned the 2D dataset quite well**. We did **not need positional embedding or UNets**. 

* Although diffusion models **work quite well** in the task of **sampling**, they tend to **perform worse** in **recreating** the overall **data density** (or finding the log-likelihood of the sample). To address this shortcoming, some authors suggested a possible way out based on the probability flow ODE[^8].

* The **learning phase is simplified** because **the transition function** $$p(x,t;x_0,0)$$ is **known explicitly**. **If** the transition function is **not known**, in paper[^9] the authors propose a loss based on a **sliced score**.

---

[^1]: $$n$$ denote the time-steps, $$l$$ is the free mean-path while $$\delta$$ is an effective collision parameter. Taken from ["Zur kinetischen Theorie der Brownschen Molekularbewegung und der Suspensionen"](https://jbc.bj.uj.edu.pl/dlibra/publication/410069/edition/386520/content) Annalen der Physik, 326, 756-780 (1906)."

[^2]: known under many different names: [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker-Planck_equation) or [forward Kolmogorov equation](https://en.wikipedia.org/wiki/Kolmogorov_equations).

[^3]: Random functions are defined with first two moments $$\left < W(t) \right > = 0$$, $$\left < W(t) W(t')\right > = \delta (t-t')$$. Time discretization $$t = n\delta t$$ makes the correlation $$\left < W(n \delta t) W(n' \delta t)\right > = \delta ( \delta t (n-n'))$$. We use the Dirac delta scaling property $$\delta (ax) = \frac{1}{a} \delta (x) $$ so that $$\delta t \left < W(n \delta t) W(n' \delta t) \right > = \delta  (n-n')$$. To retain the unit correlation after discretization, we set $$W(n \delta t) \sqrt{\delta t} = W_n$$ resulting in $$\left < W_n W_{n'} \right > = \delta_{nn'}$$.

[^4]: Because we work with images, some clipping and quantization happens so that the noise is always between [0,255]. Another minor modification of the forward process applied to images is a special form of the Langevin equation $$\frac{d}{dt} x(t) = - \frac{\beta(t)}{2} x(t) + \sqrt{\beta(t)} W(t)$$, with $\beta(t) = \beta_- + (\beta_+-\beta_-) \frac{t}{T}. This form results in initially less abrupt addition of noise.

[^5]: Derivation of this loss function is given in [this paper](https://arxiv.org/abs/2011.13456).

[^6]: ("Denoising Diffusion Probabilistic Models" Ho, Jain and Abbeel)[https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf]

[^7]: UNets are the usual architecture used in the visual modality. 

[^8]: More details are provided in [this paper](https://proceedings.mlr.press/v162/lu22f/lu22f.pdf).
[^9]: Check out [this paper](https://proceedings.mlr.press/v115/song20a.html).

