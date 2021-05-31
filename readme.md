# new idea
### the iteration process
- calculate transform matrix according to designed orbit(X=0), 
- then solve X along line, 
- then renew transform matrix according to X

so the Element class should have ability to renew its matrix according to its X

# next
verify Cr and other physical parameters, \
verify matrix

# Important

If attribute is matrix, don't change single element of matrix, but reassign the attribute.