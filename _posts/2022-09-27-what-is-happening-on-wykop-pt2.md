---
title: "What is happening on wykop.pl? pt. 2: Voting strategies"
layout: post
usemathjax: true
---

<a href="{% post_url 2022-09-27-what-is-happening-on-wykop-pt2 %}">![front](/assets/posts/2022-09-27/front2.png)</a>

This is a continuation of the exploratory work of <a href="http://wykop.pl">wykop.pl</a> I started [some time ago]({% post_url 2022-07-19-what-is-happening-on-wykop-pt1 %}). This part focuses on voting strategies. Full data can be found <a href="https://www.kaggle.com/datasets/grelade/wykop-data-2022">here</a>.



## Intro

Previously, I identified two axes of tag/author variability: tag authorship composition spans from hierarchical to egalitarian and two tag size regimes. In this part, we continue with discussing voting strategies.

The voting system on wykop.pl is similar to other social media sites. Users vote for or against particular links or findings and state the reasons behind the decision. Depending on the overall sentiment, the link can be boosted (promoted) or not (treated as regular). We find some studies[^1] investigating similar voting systems. The present study is different in two important ways a) the mechanism of promotion is unknown and b) the system is transparent and publishes plenty of data like vote timestamps and voter credentials.

Our aim is to utilize the additional data to shed light on the promotion mechanisms, identify any suspect behavior of the voters, and identify patterns in the overall voting process. We work on a subset of tags that are selected as representative in the [first part]({% post_url 2022-07-19-what-is-happening-on-wykop-pt1 %}). Each tag contains links/findings with possible duplicates. Voting data contains voting users, vote timestamps, decision rationale, and types of votes.


<!-- Previously I identified two axes of tag/author variability: tag authorship composition spans from hierarchical to egalitarian and two tag sizes. In this part, we continue with discussing the structure of votes.

The voting system on **wykop.pl** is similar to other socially-driven aggregation sites. Users vote for or against particular links and state the reasons behind the decision. Depending on the overall sentiment, the link can be boosted (promoted) or demoted (treated as regular). We find some studies[^1] which investigate similar voting systems. Our investigation is different in two important ways a) the mechanism of promotion is unknown and b) the system contains plenty of publicly available data like timestamps and voter credentials.

Our aim is to leverage this additional data and shed light on the promotion mechanisms, identify any suspect behavior of the voters and gather general information about the voting process. We work on a subset of tags singled out as representative in the [first part]({% post_url 2022-07-19-what-is-happening-on-wykop-pt1 %}). Each tag contains links/findings with possible duplicates. Voting structure contains authors, timestamps, reasons and type of vote. -->

## Global voting characteristics

We first investigate the upvote/downvote ratio for individual links separated into promoted (blue) and regular (red). Both axes are plotted on a logarithmic scale.

{:refdef: style="text-align: center;"}
![](/assets/posts/2022-09-27/fig0.png)
{: refdef}

We immediately observe a **clear cutoff line separating the groups**. We fit a simple cutoff line

<!-- $$ n_\text{upvotes} = \exp (0.05 \log(n_\text{downvotes})^2 + 4.5) $$ -->

$$ n_\text{upvotes} = 100 + 1.8 \times n_\text{downvotes} $$

which quite clearly separates promoted and regular findings. Still, although we do not find promoted findings below the curve, separation of regular links is more ambiguous as we find some regular links above the line which we call anomalies. If the cutoff curve is a boundary for the promotion-finding classifier, such cases correspond to false positives:

{:refdef: style="text-align: center;"}
![](/assets/posts/2022-09-27/fig1.png)
{: refdef}

From the point of view of any artificial voting strategy, false positives comprise interesting cases of links that did not get promoted despite acceptable upvote/downvote ratios.

### Prevalence of anomalous links

We investigate how each tag is decomposed into:

* promoted links
* regular links above the cutoff (false positives / anomalies)
* regular links below the cutoff

{:refdef: style="text-align: center;"}
![](/assets/posts/2022-09-27/fig2.png)
{: refdef}

We order them in descending order by the fraction of anomalous links. First, we observe that certain tags are more successful than others in producing promoted content. Easily understood examples of relative success are tags *#putin*, *#rosja*, *#ukraina* and *#wojna*, all treating the Ukraine war. We single out four most anomalous tags with increased fraction of false positives:

<!-- We order them in descending order by the fraction of false-positive findings. Firstly, we observe that certain tags are more successful than others in producing promoted content. Easily understood examples are *#putin*, *#rosja* and *#wojna*, all treating a recent war. On the other hand, we have a curious tag *#polska* which has an especially high fraction of false positives which we treat as an outlier due to general character of the tag (i.e. users frequently add this general tag besides the main one). We single out four tags on the left side with somewhat increased fraction of false positives: -->

- *#neuropa*
- *#energetyka*
- *#bekazpisu*
- *#wydarzenia*

These stand out both in terms of the fraction of regular links above the cutoff, and, at the same time, comprising a large fraction of promoted links.
<!-- These stand out both in terms of the fraction of regular links above the cutoff while, at the same time, comprising a large fraction of promoted links. -->
<!-- We hypothesize the false positives to be residual effects of failed upvoting campaigns resulting in not-promoted links. -->

## Types of voting trajectories

After tackling general features of votes within tags, we turn to an investigation of **time-resolved voting trajectories** for individual links. For each link, we take the whole history of upvotes/downvotes and form a curve in the upvote/downvote plane which show how an individual link gained votes during its lifetime on the website. This investigation revealed **three basic types of trajectories** depending on the relative changes in upvotes and downvotes.

<!-- We turn to investigation of time-resolved voting trajectories for single links. Based on the global upvote/downvote ratios shown above, the links themselves are divided into three groups depending on their position on the upvote/downvote plane:
* promoted
* regular above cutoff (false positives or anomalies)
* regular below cutoff -->

<!-- Each link has its own voting trajectory, we identify three basic types
* hockey - voting trajectory hovers at first below the cutoff line and then gains a sudden momentum and shoots up above the cutoff line
* linear - both types of votes increase in approximately constant proportion independent of the overall popularity of the finding and whether it crosses the cutoff line

* inverse hockey - voting trajectory has high momentum from the beginning and only later saturates -->

* **hockey** - initially, the voting trajectory hovers below the cutoff line and then gains an additional upvote momentum and shoots up above the cut-off line.

* **linear** - both up- and down-votes increase in approximately constant proportion throughout the links' lifetime; the crossing through the cutoff line does not affect this dynamic.

* **inverse hockey** - initially, upvotes dominate over downvotes resulting in a fast crossing of the cutoff line, which only later saturates

On the same upvote/downvote plane as before but with linear scales, we present instances of all three trajectory types.

{:refdef: style="text-align: center;"}
![](/assets/posts/2022-09-27/fig4.png)
{: refdef}

These types mirror the general popularity of the finding. In particular, we observe inverse-hockey types mainly concerning hot topics like the Ukraine war. Among all studied links, the most numerous are the hockey types while we find only a few linear cases. The **hockey** types are particularly interesting as there exists **a time-localized impulse** changing the voting dynamics and moving the finding above the cutoff line. Unfortunately, we cannot tell whether the impulse is a side-effect of the promotion being toggled or rather as a result of some external factor. The former explanation is in conflict with the presence of hockey-type trajectories in the anomalous link group. Still, this cannot be resolved since we do not have access to historical promotion data.

<!-- These types represent the overall popularity of the finding. In particular, we observe inverse-hockey types in hot topics like the ukraine war. The most probable is the hockey type with few linear cases which show constant popularity upon growth. The hockey types are particularly interesting as there exist an impulse changing the voting dynamics and moving the finding above the cutoff line. Unfortunately we cannot tell whether this impulse is the side-effect of the promotion or rather as a result of an organized action. The former explanation is in contradiction with the presence of hockey-type trajectories in the regular-above-line link group. Still, this cannot be resolved since we do not have access to historical data on promotions.  -->

<!-- In general, the regular-above-line case contains similar types of trajectories, we cannot make a distinction based on the trajectories: -->

Inspection of trajectories in the promoted and anomalous/regular-above-cutoff link groups does not reveal any differences.

{:refdef: style="text-align: center;"}
![](/assets/posts/2022-09-27/fig5.png)
{: refdef}

<!--

In the plot above we show instances of major trajectory classes:
- publication time peak, which comprise of a single peak of upvotes around the publication time. Typically it is localized.
- late-time peak, which shows up later on in the vote evolution. It might be an indicator of a coordinated upvoting campaign.
- -->

## Conclusions

In this part, I looked at global voting characteristics. The main conclusions are as follows.

- In the upvote/downvote plane, there exists a **simple linear cutoff line separating promoted and regular links**
- **Presence of anomalies**, i.e. regular links, which, according to the cutoff line, should be promoted
- **Three types of voting trajectories**: hockey, linear or inverse-hockey depending on the dynamics of vote gains
- Types of **voting trajectories are qualitatively the same** within promoted and anomalous link groups

In the last part I plan to focus on informal user groups.

<!-- In this part, I looked at global voting characteristics. Main conclusions are

- In the upvote/downvote plane, I identified a simple **linear cutoff line** separating promoted and regular links**
- Presence of **anomalies** i.e. regular links which, according to the cutoff line, should be promoted
- Types of voting strategies are **hockey, linear or inverse-hockey** depending on how they gain votes
- Considering promoted and regular-above-cutoff link groups, **no differences in voting trajectories** were found

In the last part I plan to focus on informal user groups. -->

[^1]: *[ Identifying and Quantifying Coordinated Manipulation of Upvotes and Downvotes in Naver News Comments ](https://ojs.aaai.org/index.php/ICWSM/article/view/7301)*
