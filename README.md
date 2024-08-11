# Birds! 

I wanted to create a bird classifier project because I like birds & machine learning! 

I am learning so much from my study group for Andrej Karpathy's Zero to Hero series! I'm getting a great overview of how neural networks work in general and revisited some of these topics even with this quick project! 


Here's how I did this: 

- there's actually a huggingface pre-trained library for birds! So Easy - https://huggingface.co/chriamue/bird-species-classifier . Looks like it was created 9 months ago (Nov 2023) - crazy how fast ML is moving - so exciting. 


### Demo

- as of 8/10/2024, this project needs some improvement! Check out my demo here : 

[![YouTube Video](https://img.youtube.com/vi/cgtEztkR0NY/0.jpg)](https://youtu.be/cgtEztkR0NY)


1. Are you a better bird watcher than this huggingface model? Prove it- reach out and show us which ones are wrong! 

2. Any ideas for how to fine tune or otherwise improve this model to be better at Florida birds?? 


### Pics
- My best friend Andrea is a biologist in Florida and takes the best bird pictures. I just gathered some off my phone to use for this project, but Andrea was the one who took most of them - so thanks Andrea! 

- Follow her on insta to stay up to date on the bird pics (@andrealfuchs)



### Software 

I'm a fan of pipenv to manage my dependencies. 

Here's your dependencies: 

- pipenv install : 
- transformers
- torch
- torchvision
- Pillow (in the code you'll see it as PIL, but PIL didn't work for me, I had to do Pillow)
- streamlit

here's that all together
```
pipenv install transformers torch torchvision Pillow streamlit
```

### How to run 
`streamlit run birds.py`

- upload a png file 
- check out my folder "bird_pics" for some samples
- you can also switch it to jpg or other file types by modifying the code slightly 


### Things I learned
- RGB vs RBGA - Andrea's camera is fancy and uses RGBA (alpha channel)
- I had to add a line of code 
```
if image.mode != 'RGB':
    image = image.convert('RGB') 
```
- this puts all the images into RGB 

### To Do 
*stuff I want to do to improve this*
- [ ] feed it videos
- [ ] how to improve this model so it is better with florida birds? 
- ✅ give it a drag and drop 
interface and deploy it so other people can use it   - Done! Thanks Streamlit

