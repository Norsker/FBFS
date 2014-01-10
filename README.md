FBFS
====


Algorithm description
====

CMIM.py is an implementation of the feature selection algorithm described in http://machinelearning.wustl.edu/mlpapers/paper_files/Fleuret04.pdf.  For a binary classification problem, it selects a subset of the (binary) predictors based on conditional mutual information maximization.

http://en.wikipedia.org/wiki/Conditional_mutual_information

The paper's pseudocode for the simplest version of the algorithm is as follows:

N=total features
K=number of selected features
```
for n = 1...N do	s[n] ← mut_inf(Y,n)for k = 1...K do	nu[k] = arg max (n) s[n] for n = 1...N do	s[n] ← min(s[n],conditional_mut_inf(Y,n | nu[k]))
```

	
In my understanding, there seems to be an omission in the paper: since K is not preselected, there needs to be a termination condition that activates when there are no features that add information conditional on the already selected features, and we iterate on the already selected K' variables, not K.

The 'fast' implementation of this algorithm takes advantage of the fact that a score s[n] can only decrease as the algorithm iterates.  Consequently, we do not need to calculate an updated score s[n] for feature n if it is lower than the best up-to-date score for features 1,..,n-1.  The given fast version pseudocode is:

```for n = 1...N do	ps[n] ← mut_inf(Y,n) 
	m[n] ← 0for k = 1...K do
 s* ← 0	for n = 1...N do		while ps[n]>s* and m[n] <k−1 do			m[n] ← m[n]+1			ps[n] ← min(ps[n],cond_mut_inf(Y,n,nu[m[n]]))
		 if ps[n] > s* then			s* ← ps[n] nu[k] ← n
```
If most features are irrelevant or redundant, this cuts down on a lot of iterations.



Implementation Notes
====

The joint entropy calculation is based off of the implementation at
http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/.

def entropy(*X):
    return(np.sum(-p * np.log2(p) if p > 0 else 0 for p in
        (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
            for classes in itertools.product(*[set(x) for x in X]))))

The set() operation in itertools.product, while elegantly allowing for the random variables taking any number of values, was very computationally expensive.  Since I know, however, that all inputs to this algorithm should be binary, I replaced it with *[(0,1) for x in range(0,n_elements)]; for even small samples(1000 observations, 3 R.V.) this was about 450 times faster.

The primary bottleneck for dense input is that the cond inf/entropy calculation uses too much python, and should probably be replaced with cython/smarter numpy calls(i.e., using numpy's reduce instead of python's reduce).

That said, the primary problem right now is the handling of sparse input; I'm going to need to figure out how to properly calculate entropy without leaving csc format, because converting the selected columns to dense format in the calculation of conditional information unsurprisingly adds an enormous amount of computational time.  In the short term, I could alter the conditional_inf function to take X,y instead of X, and keep the y vector stored in dense format from the beginning, since y is used in every joint entropy calculation.  

