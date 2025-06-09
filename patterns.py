def lower_triangular(n):
    for i in range(1, n+1):
        print('*' * i)
# calling lower triangular pattern function
lower_triangular(5)
print("\n")

def upper_triangular(n):
    for i in range(n):
        print(' '*i + '*'*(n-i))
# calling upper triangular pattern function
upper_triangular(5)
print("\n")

def pyramid(n):
    for i in range(1, n+1):
        print(' '*(n-i) + '*'*(2*i-1))
# calling pyramid pattern function
pyramid(5)