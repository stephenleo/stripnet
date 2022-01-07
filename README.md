# 💡 STriP Net: Semantic Similarity of Scientific Papers (S3P) Network

[![DOI](https://zenodo.org/badge/444768334.svg)](https://zenodo.org/badge/latestdoi/444768334)

Do you read a lot of Scientific Papers? Have you ever wondered what are the overarching themes in the papers that you've read and how all the papers are semantically connected to one another? Look no further!

Leverage the power of NLP Topic Modeling, Semantic Similarity, and Network analysis to study the themes and semantic relations within a corpus of research papers.

✅ Generate STriP Network on your own collection of research papers with just three lines of code!

✅ Interactive plots to quickly identify research themes and most important papers

✅ This repo was hacked together over the weekend of New Year 2022. This is only the initial release, with lots of work planned.

💪 Please leave a ⭐ to let me know that STriP Net has been helpful to you so that I can dedicate more of my time working on it.

## ⚡ Install
- Highly recommend to install in a conda environment
```
conda create -n stripnet python=3.8 jupyterlab -y
conda activate stripnet
```

- Pip install this library
```
pip install stripnet
```

## 🔥🚀 Generate the STriP network analysis on default settings
- STriP can essentially run on any pandas dataframe column containing text. 
- However, the pretrained model is hardcoded (for now), so you'll see the best results while running it on a column that combines the `title` and `abstract` of papers separated by `[SEP]` keyword. Please see below 

```
# Load some data
import pandas as pd
data = pd.read_csv('data.csv')

# Keep only title and abstract columns
data = data[['title', 'abstract']]

# Concat the title and abstract columns separated with [SEP] keyword
data['text'] = data['title'] + '[SEP]' + data['abstract']
```

```
# Instantiate the StripNet
from stripnet import StripNet
stripnet = StripNet()

# Run the StripNet pipeline
stripnet.fit_transform(data['text'])
```

- If everything ran well, your browser should open a new window with the network graph similar to below. The graph is fully interactive! Have fun playing around by hovering over the nodes and moving them around!
- If you are not satisfied with the topics you get, just restart the kernel and rerun it. The Topic Modeling framework has some level of randomness so the topics will change slightly with every run.
- You can also tweak the paremeters of the various models, please look out for the full documentation for the details!

![STriP Network](https://github.com/stephenleo/stripnet/blob/main/images/strip_network.png?raw=true "Sample STriP Network")

## 🏅 Find the most important paper
- After you fit the model using the above steps, you can plot the most important papers with one line of code
- The plot is fully interactive too! Hovering over any bar shows the relevant information of the paper.

```
stripnet.most_important_docs()
```

![Most Important Text](https://github.com/stephenleo/stripnet/blob/main/images/centrality.png?raw=true "Most Important Papers")

## 🛠️ Common Issues
1. If your StripNet graph is just one big ball of moving fireflies, try these steps
    - Check the value of `threshold` currently used by stripnet
    ```
    current_threshold = stripnet.threshold
    print(current_threshold)
    ```
    - Increase the value of `threshold` in steps of 0.05 and try again until you see a good looking network. Remember the max value of threshold is 1! If you're threshold is already 0.95 then try increasing in steps of 0.01 instead.
    ```
    stripnet.fit_transform(data['text'], threshold=current_threshold+0.05)
    ```
 2. If you're dataset is small (<500 rows) and the number of topics generated seems too less
    - Try tweaking the value of `min_topic_size` to a value lower than the default value of 10 until you get topics that look reasonable to you
    ```
    stripnet.fit_transform(data['text'], min_topic_size=5)
    ```   
 3. After the above two steps, if your graph looks messy, try removing isolated nodes (those nodes that don't have any connections)
    ```
    stripnet.fit_transform(data['text'], remove_isolated_nodes=True)
    ```  
 4. In practice, you might have to tweak all three at the same time!
    ```
    stripnet.fit_transform(data['text'], threshold=current_threshold+0.05, min_topic_size=5, remove_isolated_nodes=True)
    ```
        
 I'm testing out the network on a variety of data to pick better default values. Do let me know if some specific values worked the best for you!

## 🎓 Citation
To cite STriP Net in your work, please use the following bibtex reference:
```
@software{marie_stephen_leo_2022_5823822,
  author       = {Marie Stephen Leo},
  title        = {STriP Net: Semantic Similarity of Scientific Papers (S3P) Network},
  month        = jan,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.0.5.zenodo},
  doi          = {10.5281/zenodo.5823822},
  url          = {https://doi.org/10.5281/zenodo.5823822}
}
```

## 🤩 Acknowledgements
STriP Net stands on the shoulder of giants and several prior work. The most notable being
1. Sentence Transformers [[Paper]](https://arxiv.org/abs/1908.10084) [[Code]](https://www.sbert.net/)
2. AllenAI Specter pretrained model [[Paper]](https://arxiv.org/abs/2004.07180) [[Code]](https://github.com/allenai/specter)
3. BERTopic [[Code]](https://github.com/MaartenGr/BERTopic)
4. Networkx [[Code]](https://networkx.org/)
5. Pyvis [[Code]](https://github.com/WestHealth/pyvis)

## 🙏 Buy me a coffee
If this work helped you in any way, please consider the following way to give me feedback so I can spend more time on this project
1. ⭐ this repository
2. ❤️ [the Huggingface space ](https://huggingface.co/spaces/stephenleo/strip) (Coming Jan 11 2022!)
3. 👏 [the Medium post](https://stephen-leo.medium.com/) (Coming End Jan 2022!)
4. ☕ [Buy me a Coffee!](https://www.buymeacoffee.com/stephenleo)
