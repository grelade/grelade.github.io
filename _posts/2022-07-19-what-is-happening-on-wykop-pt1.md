---
title: "What is happening on wykop.pl? pt. 1: Authors"
layout: post
---

<a href="{% post_url 2022-07-19-what-is-happening-on-wykop-pt1 %}">![front](/assets/posts/2022-07-19/front2.png)</a>

For some time I wondered about the inner workings of <a href="http://wykop.pl">wykop.pl</a>, one of Poland's main social networks not related to GAFA behemoths. In particular, there is an ongoing discussion about the existence and character of vote brigading campaigns on the website. Full data can be found <a href="https://www.kaggle.com/datasets/grelade/wykop-data-2022">here</a>.



## Background

Wykop is a Polish community-based aggregation site similar to <a href="http://reddit.com">Reddit</a> or <a href="http://digg.com">Digg</a>. Since its launching in 2005, it had at least a few changes when it comes to its overall character and demographics. It started out as a mecca for young IT workers, filled with elitist humor and high-quality content. With increasing popularity and a flood of new users, this geeky site gained an increasingly political flavor. Around 2015, there was a surge of suspicious activities aligned with the political campaign of the KORWiN libertarian party and centered around Janusz Korwin-Mikke, its most prominent leader. As a relatively large social medium, the site became a tool in the elections and, although KORWIN finally did not get elected, the online campaign itself was considered a success. After such seemingly grass-root campaigns organized by political activists, in the following elections, the dominant parties PiS and PO involved PR/marketing firms to similar types of campaign. These developments occurred largely in the context of the Trump election, the Brexit vote, and the general discussion around Cambridge Analytica. From thereon, the site becomes a battleground of influence wars between various groups. Today, some users report on a nebulous neuropa group whose activities on the website are considered a liberal-left reaction to previous right-leaning movements.

## Motivation

I am looking for any out-of-ordinary behavior in the user's activity on the wykop website. My aim is to identify:

- informal groups of influence
- upvoting campaigns
- botnets

## Approach and dataset

To collect data, I built a <a href="http://github.com/grelade/wykop-scrap">simple scraping engine</a> in Python. I looked at more than seven months of data between 2022-01-01 and 2022-07-15, covering the Ukraine conflict. Since there are many possible ways to penetrate this dataset, as a first approach, I focus on the most popular tags, expand the dataset by first building, for each tag, a database of upvoted (hot) links. I build an increasingly complex set in three dimensions,

* **global view for all tags**
    - number of authors vs. number of upvoted articles
    - global upvotes / downvotes vs. number of upvoted articles
    - interconnectivity of tags
* **local characteristics for single tag**
    - composition of most successful authors
    - upvoting / downvoting structure
    - temporal voting structure
* **characteristics for a user**

Importantly, for now, I ignore the mikroblog activity, which is a less formal form of communication that forms a large chunk of websites' activity. Data is available publicly <a href="https://www.kaggle.com/datasets/grelade/wykop-data-2022">here</a>.

## Global view

From a bird's eye view, tags are subcommunities whose properties are to be inspected statistically. Each tag has an activity parameter measured by the number of upvoted links and a size-like value measured by the number of authors creating the links. These two are plotted for each tag on a log-scale:

{% include_relative 2022-07-19/fig1.html %}

First, we are able to **identify two size regimes**. For **small tags** with a number of links less than 100, there is a sizeable number of tags with a very **flat distribution of authors** where the number of upvoted links is almost equal to the number of authors; there is **no hierarchy of active users** within the tag.

This linear relation breaks down for **larger tags**, where there are still relatively homogeneous tags like *#heheszki* with an average user contributing with roughly two links, but a clear departure for the largest tags like *#ukraina* can reach a user-to-link ratio of 1:7. In such cases, there seems to be **a popularity incentive** where certain **users tend to become more active** but only when the tag becomes en vogue.

Besides the bulk of tags, there are clear **outliers** whose aforementioned **user-to-link ratio is high** as encoded by the point sizes in figure above. Such a simple indicator suggests that there are certain **tags with relatively small groups of very active and successful users**.

Based on this analysis, I identify **two axes of variability**:

* **activity of the tag** (small vs. large number of upvoted links)
* **user participation** (egalitarian vs. hierarchical author structure)


### Author structure
I find large hierarchical communities by listing the top 10 most active tags (more than 100 upvoted links) and sort them by user-to-link ratio:

{% include_relative 2022-07-19/tab1.html %}

I inspect the author structure by user-to-link ratio and through a Gini coefficient. Figures below show that among the largest tags (with more than 250 links), **tendencies towards hierarchical and egalitarian authorship structure are visible**. We look at weekly fraction of best links partitioned among the top 10% of the authors.

In **hierarchical cases**, we observe a minority of authors dominating successful link creation within the tag with above 50% of the total number of links. There are examples of both smaller tags like *#dobrazmiana* and *#liganauki* where this effect is extreme and sometimes reaches 100% however we find also an outlier *#wydarzenia* which is both large and extremely dominated by only few users.

{% include_relative 2022-07-19/fig2.html %}

We turn to **egalitarian tags** which tend to be smaller in size than their hierarchical counterparts (mean sizes are 312 and 623 for the egalitarian and hierarchical tags, respectively). There is also an overall tendency for these tags to be non-political and interest-oriented.

{% include_relative 2022-07-19/fig3.html %}

To clearly present discussed distinction, we show authorship composition of a hierarchical tag *#wydarzenia* and egalitarian tag *#technologia*.

The ***#wydarzenia* tag** shows clearly a **hegemony of users *Szewczenko* and *Szu_*** which both comprise almost half of all succesful articles on the tag.

{% include_relative 2022-07-19/fig4-wydarzenia.html %}

On the other hand, there is **no clear winner** in the ***#technologia* tag**, as top 10 authors do not dominate the tag output.

{% include_relative 2022-07-19/fig4-technologia.html %}

### *#neuropa* tag

On the basis of this analysis, we lastly show the aforementioned ***#neuropa* tag**. It has a relatively **hierarchical character** but **lacks a clear leader** like the duo in the *#wydarzenia* tag. In the number of authors/ number of links plot, it is present on an outlier position.

{% include_relative 2022-07-19/fig5-neuropa.html %}


### Conclusions

In this part, I inspect the global properties of user activity organized within subcommunities around the most popular tags on the social media website wykop.pl. I found:

- **two size regimes** in tags
- existence of **outliers**
- tags have either **hierarchical or egalitarian** structure when considering users' activity where a minority of users can have unparalleled influence

In the second part, I plan to inspect what are the voting structure within tags.
