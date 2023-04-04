# MiniProject
Some days ago, I read a paper named "Enhancing Sequential Recommendation with Graph Contrastive Learning​​​​​. " The author proposed a method to encode users' historical behavior sequence. 
The sequence is similar to the path in ROAS. So I used this method to build a global graph to encode the path and calculate the weight between different devices.
After that, I convert every path into weights. Then I used the non-null data to train a regression model and generate pseudo labels for null data.
Lastly, I use every data to train the model again.

I provide different version of my report including:
- ROAS.html
- ROAS.ipynb
- ROAS.md
- ROAS.pdf
