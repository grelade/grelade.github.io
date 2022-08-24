---
title: "What is happening on wykop.pl? pt. 1"
layout: post
---

![front](/assets/posts_img/2022-07-19.png)

For some time I wondered about the inner workings of <a href="http://wykop.pl">wykop.pl</a>, one of the largest non-GAFA-related social medium in Poland. In particular, there is an ongoing discussion about existence and character of upvoting campaigns on the website. 


## Background

Wykop is a polish community-based aggregation site similar to <a href="http://reddit.com">reddit</a> and <a href="http://digg.com">digg</a>. Since it's launching in 2005, it had at least a few turns when it comes to its overall character and demographics. It started off as a mecca for young IT workers, filled with elitist humor and high-quality content. With increasing popularity and a flood of new users, this geeky site gained an incereasingly political flavor. Around 2015, there was a surge of suspicious activities synced with political campaign of the KORWiN libertarian party and centered around its most prominent leader Janusz Korwin-Mikke. As a relatively large social medium, the site became a tool in the elections and, although KORWIN finally did not get elected, the online campaign itself was considered a success. After such seemingly grassroot campaigns organized by political activists, in the following elections the dominant parties PiS and PO involved PR/marketing firms to similar types of campaign. These developments were happening largely in the context of the Trump election, Brexit vote and the overall discussion around Cambridge Analytica. From thereon, the site becomes a battleground of influence wars between various groups. Nowadays, some users report on a nebulous *neuropa* group whose activities on the website are considered a liberal-left reaction to the previous right-leaning movements.

## Motivation

I am looking for any out-of-ordinary behaviour in the user's activity on the wykop website. My aim is to identify:
- informal groups of influence
- upvoting campaigns 
- botnets

## Approach and dataset

To gather data I built a <a href="http://github.com/grelade/wykop-scrap">simple scraping engine</a> in python. I look at over **seven months of data** between 2022-01-01 and 2022-07-15 which covers Ukraine conflict. Since there are many possible ways to penetrate this data, as a first approach I focus on the most popular tags and expand the dataset from there by first building a database of upvoted (hot) links and their respective authors for each tag. I inspect an increasingly complex set in three dimensions:
* **global view for all tags**
    - number of authors vs. number of upvoted articles
    - global upvotes / downvotes vs. number of upvoted articles
    - interconnectivity of tags
* **local characteristics for single tag**
    - composition of most successful authors
    - upvoting / downvoting structure
    - temporal voting structure
* **characteristics for a user**

Importantly, for now I ignore the *mikroblog* activity which is a less formal form of communication forming a sizeable chunk of websites' activity.

## Global view

From a bird's eye view, tags are sub-communities whose properties are to be inspected statistically. Each tag has an activity parameter measured by the number of upvoted links and a size-like value measured by the number of authors creating the links. These two are plotted for each tag on a log scale:

{% include_relative 2022-07-19/fig1.html %}

Firstly, we are able to **identify two size regimes**. For **small tags** with number of links <100, there is a sizeable number of tags with a very **flat distribution of authors** where the number of upvoted links is almost equal to the number of authors; there is **no hierarchy of active users** within the tag. 

This linear relation breaks down for **larger tags** where still there are relatively homogeneous tags like *#heheszki* with an average user contributes with roughly two links but a clear departure for the largest tags like *#ukraina* can reach a user-to-link ratio of 1:7. In such cases, there seems to be **a popularity incentive** where **certain users tend to become more active** but only when the tag becomes en vogue. 

Besides described bulk of tags, there are clear **outliers** whose aforementioned **user-to-link ratio is high** as encoded by the point sizes in figure above. Such a simple indicator suggests that there are certain **tags with relatively small groups of very active and succesful users**. 

Based on this analysis, I identify **two axes of variability**:
* **activity of the tag** (small vs. large number of upvoted links)
* **user participation** (egalitarian vs. hierarchical author structure)


### Author structure
I find large hierchical communities by listing top 10 most active tags (more than 100 upvoted links) and sort them by user-to-link ratio:

{% include_relative 2022-07-19/tab1.html %}

I inspect the author structure by user-to-link ratio and through a Gini coefficient. Figures below show that among the largest tags (with >250 links), **tendencies towards hierarchical and egalitarian authorship structure are visible**. We look at weekly fraction of best links partitioned among the top 10% of the authors. 

In **hierarchical cases**, we observe a minority of authors dominating succesful link creation within the tag with above 50% of the total number of links. There are examples of both smaller tags like *#dobrazmiana* and *#liganauki* where this effect is extreme and sometimes reaches 100% however we find also an outlier *#wydarzenia* which is both large and extremely dominated by only few users.

{% include_relative 2022-07-19/fig2.html %}

We turn to **egalitarian tags** which tend to be smaller in size than their hierarchical counterparts (mean sizes are 312 and 623 for the egalitarian and hierchical tags respectively). There is also an overall tendency for these tags to be non-political and interest oriented.

{% include_relative 2022-07-19/fig3.html %}

To clearly present discussed distinction, we show authorship composition of a hierarchical tag *#wydarzenia* and egalitarian tag *#technologia*. 

The ***#wydarzenia* tag** shows clearly a **hegemony of users *Szewczenko* and *Szu_*** which both comprise almost a half of all succesful articles on the tag. 

{% include_relative 2022-07-19/fig4-wydarzenia.html %}

On the other side, there is **no clear winner** in the ***#technologia* tag** as top 10 authors do not dominate the tag output.

{% include_relative 2022-07-19/fig4-technologia.html %}

### *#neuropa* tag

Based on this analysis, we lastly show the aforementioned ***#neuropa* tag**. It has a relatively **hierarchical character** but **lacks a clear leader** alike the duo in the *#wydarzenia* tag. In the number of authors/ number of links plot it is present on an outlier position.

{% include_relative 2022-07-19/fig5-neuropa.html %}


### Conclusions

In this part I inspected global properties of user activity organized within subcommunities around most popular tags on the social media website wykop.pl. I found:

- **two size regimes** in tags
- existence of **outliers**
- tags have either **hierarchical or egalitarian** structure when considering users' activity where a minority of users can have unparalleled influence

In the second part I plan to inspect further whether these dominant actors utilize special tactics to gain influence. 