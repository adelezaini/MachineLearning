# Template for evaluating and grading projects, with grading scale based on achieved points

**Evaluation of project number: 131** 
**Name: Adele Zaini** 

## Abstract 
*Abstract: accurate and informative? Total number of possible points: 5* 
Mark and comments: The abstract is nice, and it explains the main findings of the project. Good job 5/5 

## Introduction 
*Introduction: status of problem and the major objectives. Total number of possible points: 10* 
Mark and comments: You give a brief introduction to the topic of neural networks, and state what problems will be explored in the report. Feed forward neural networks, SGD, optimization of hyperparameters, comparison to regression is all mentioned. Very nice intro, good job! **One comment to make it even more like a paper would be to give some sources for the historical background.** This will not detract anything just a comment. 10/10 

## Formalism 
*Formalism/methods: Discussion of the methods used and their basis/suitability. Total number of possible points 20* 
Mark and comments: I personally think the Methods section **should be split into Theory and Methods**. It reads a bit cleaner than having Methods and then an immediate subsection after. Nevertheless the discussion of the theory is excellent! **Are the figures made by you? If so they are great, if not they should be sourced!** Great explanation of back propagation. The discussion of activation functions I find a bit lacking. Only one of the three functions asked to be studied is written up. Metrics and accuracy is nice. Very good work on the discussion of gradient descent and SGD! Again, is this figure your own work or someone else's? The figure for the comparison is excellent. Very thorough discussion of the various optimizers. 18/20 

## Code, implementation and testing 
*Code/Implementations/test: Readability of code, implementation, testing and discussion of benchmarks. Total number of possible points 20* 
Mark and comments: Super nice job with the Jupyter notebooks not being too long. This often makes them unbearable to work with. Nice job splitting the code up into separate files and importing functions when they are necessary. ++ I also like the presentation of your functions in the text, good job, but **specific implementation details fit better in the github repository readme or docstrings.** Your neural network routine is a very good example of how method should be presented. Makes it simple to get an overview over everything you have done. There is not much else to comment on here in the method section, very nice work! 18/20 

## Analysis 
*Analysis: of results and the effectiveness of their selection and presentation. Are the results well understood and discussed? Total number of possible points: 20* 
Mark and comments: You present the results for your learning schedule analysis well, but I don't really see **any discussion of the results beyond listing what you got.** In my opinion it **should be clearer which dataset the analysis is on**. Instead of Task a - SGD, it would read better just having Franke Function there. Minor detail. For the Feedforward neural network part, it would have been better if you explained these unexpected problems, **why you think they're problems, potential error sources.** Good comparison of activation functions. Nice addition speaking of the vanishing gradient problem. I feel **sources for further reading** etc should have been included here.**Very nice that you always write how to reproduce your findings by including all the relevant parameters used to generate the plots.** You lose out by lacking the logistic regression part here. 11/20 

## Conclusions 
*Conclusions, discussions and critical comments: on what was learned about the method used and on the results obtained. Possible directions and future improvements? Total number of possible points: 10* 
Mark and comments: It is stated that we can appreciate its better performance to linear regression. Why might this be the case? Is it reasonable to expect this for both our data sets? I feel the conclusions are a bit lacking. This was a whole task in the project and only three small paragraphs consisting mostly of re-listing the results from before. I won't fault too much here, but **there should have been more mention of: the pros and cons of each algorithm compared to each other**. 7/10 

## Overall presentation: 
*Clarity of figures, tables, algorithms and overall presentation. Too much or too little? Total number of possible points: 10* 
Mark and comments: **You should not put the problem descriptions in the project text**. 7/10 

## Referencing 
*Referencing: relevant works cited accurately? Total number of possible points 5* 
Mark and comments: There is **no usage of citations in the text, they are only found at the bottom**. 2/5 

## Overall 
*Overall mark in points (maximum number of points per project is 100) and final possible final comments* 78/100 

## Grading of all projects 
*The final number of points is based on the average of all projects (including eventual additional points) and the grade follows the following table:* * 92-100 points: A * 77-91 points: B * 58-76 points: C * 46-57 points: D * 40-45 points: E * 0-39 points: F-failed

##  General guidelines on how to write a report

### Some basic ingredients for a successful numerical project

When building up a numerical project there are several elements you should think of, amongst these we take the liberty of mentioning the following:

 *   How to structure a code in terms of functions
 *   How to make a module
 *   How to read input data flexibly from the command line
 *   How to create graphical/web user interfaces
 *   How to write unit tests (test functions)
 *   How to refactor code in terms of classes (instead of functions only), in our case you think of a system and a solver class
 *   How to conduct and automate large-scale numerical experiments
 *   How to write scientific reports in various formats (LaTeX, HTML)


The conventions and techniques outlined here will save you a lot of time when you incrementally extend software over time from simpler to more complicated problems. In particular, you will benefit from many good habits:

 * New code is added in a modular fashion to a library (modules)
 * Programs are run through convenient user interfaces
 * It takes one quick command to let all your code undergo heavy testing
 * Tedious manual work with running programs is automated,
 * Your scientific investigations are reproducible, scientific reports with top quality typesetting are produced both for paper and electronic devices.




### The report: how to write a good scienfitic/technical report
What should it contain? A typical structure

* An abstract where you give the main summary of your work
 * An introduction where you explain the aims and rationale for the physics case and  what you have done. At the end of the introduction you should give a brief summary of the structure of the report
 * Theoretical models and technicalities. This is the methods section
 * Results and discussion
 * Conclusions and perspectives
 * Appendix with extra material
 * Bibliography

Keep always a good log of what you do.

### The report, the abstract

The abstract gives the reader a quick overview of what has been done and the most important results. Try to be to the point and state your main findings. It could be structured as follows
o Short introduction to topic and why its important
o Introduce a challenge or unresolved issue with the topic (that you will try to solve)
o What have you done to solve this
o Main Results
o The implications




### The report, the introduction

When you write the introduction you could focus on the following aspects

 * Motivate the reader, the first part of the introduction gives always a motivation and tries to give the overarching ideas
 * What I have done
 * The structure of the report, how it is organized etc

### The report, discussion of methods, implementation, codes etc

 * Describe the methods and algorithms
 * You need to explain how you implemented the methods and also say something about the structure of your algorithm and present some parts of your code
 * You should plug in some calculations to demonstrate your code, such as selected runs used to validate and verify your results. The latter is extremely important!!  A reader needs to understand that your code reproduces selected benchmarks and reproduces previous results, either numerical and/or well-known  closed form expressions.



### The report, results part

 * Present your results
 * Give a critical discussion of your work and place it in the correct context.
 * Relate your work to other calculations/studies
 * An eventual reader should be able to reproduce your calculations if she/he wants to do so. All input variables should be properly explained.
 * Make sure that figures and tables should contain enough information in their captions, axis labels etc so that an eventual reader can gain a first impression of your work by studying figures and tables only.

### The report, conclusions and perspectives

 * State your main findings and interpretations
 * Try as far as possible to present perspectives for future work
 * Try to discuss the pros and cons of the methods and possible improvements


### The report, appendices

 * Additional calculations used to validate the codes
 * Selected calculations, these can be listed with  few comments
 * Listing of the code if you feel this is necessary
 
You can consider moving parts of the material from the methods section to the appendix. You can also place additional material on your webpage or GitHub page.. 

### The report, references

 * Give always references to material you base your work on, either  scientific articles/reports or books.
 * Refer to articles as: name(s) of author(s), journal, volume (boldfaced), page and year in parenthesis.
 * Refer to books as: name(s) of author(s), title of book, publisher, place and year, eventual page numbers
