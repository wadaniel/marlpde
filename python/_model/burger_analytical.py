def burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt, f0 = None):

#*****************************************************************************80
#
## burgers_viscous_time_exact1() evaluates a solution to the Burgers equation.
#
#  Discussion:
#
#    The form of the Burgers equation considered here is
#
#      du       du        d^2 u
#      -- + u * -- = nu * -----
#      dt       dx        dx^2
#
#    for -1.0 < x < +1.0, and 0 < t.
#
#    Initial conditions are u(x,0) = - sin(pi*x).  Boundary conditions
#    are u(-1,t) = u(+1,t) = 0.  The viscosity parameter nu is taken
#    to be 0.01 / pi, although this is not essential.
#
#    The authors note an integral representation for the solution u(x,t),
#    and present a better version of the formula that is amenable to
#    approximation using Hermite quadrature.
#
#    This program library does little more than evaluate the exact solution
#    at a user-specified set of points, using the quadrature rule.
#    Internally, the order of this quadrature rule is set to 8, but the
#    user can easily modify this value if greater accuracy is desired.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Claude Basdevant, Michel Deville, Pierre Haldenwang, J Lacroix,
#    J Ouazzani, Roger Peyret, Paolo Orlandi, Anthony Patera,
#    Spectral and finite difference solutions of the Burgers equation,
#    Computers and Fluids,
#    Volume 14, Number 1, 1986, pages 23-41.
#
#  Input:
#
#    real NU, the viscosity.
#
#    integer VXN, the number of spatial grid points.
#
#    real VX(VXN), the spatial grid points.
#
#    integer VTN, the number of time grid points.
#
#    real VT(VTN), the time grid points.
#
#  Output:
#
#    real VU(VXN,VTN), the solution of the Burgers
#    equation at each space and time grid point.
#
  import numpy as np

  qn = 8
#
#  Compute the rule.
#
  qx, qw = hermite_ek_compute ( qn )
#
#  Evaluate U(X,T) for later times.
#
  vu = np.zeros ( [ vxn, vtn ] )
      
  if (f0 is None):
    f0 = lambda x: -np.sin( np.pi * x)
   

  vu[:, 0] = f0(vx)

  for vti in range ( 1, vtn ):


      for vxi in range ( 0, vxn ):

        top = 0.0
        bot = 0.0

        for qi in range ( 0, qn ):

          c = 2.0 * np.sqrt ( nu * vt[vti] )

          top = top - qw[qi] * c * np.sin ( np.pi * ( vx[vxi] - c * qx[qi] ) ) \
            * np.exp ( - np.cos ( np.pi * ( vx[vxi] - c * qx[qi]  ) ) \
            / ( 2.0 * np.pi * nu ) )

          bot = bot + qw[qi] * c \
            * np.exp ( - np.cos ( np.pi * ( vx[vxi] - c * qx[qi]  ) ) \
            / ( 2.0 * np.pi * nu ) )

          vu[vxi,vti] = top / bot

  return vu

def burgers_viscous_time_exact1_test01 ( ):

#*****************************************************************************80
#
## burgers_viscous_time_exact1_test01() tests sets up a small test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  vtn = 11
  vxn = 11
  nu = 0.01 / np.pi

  print ( '' )
  print ( 'burgers_viscous_time_exact1_test01():' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  burgers_viscous_time_exact1() evaluates solution #1' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = -1.0
  xhi = +1.0
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 3.0 / np.pi
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

  r8mat_print ( vxn, vtn, vu, '  U(X,T) at grid points:' )

  filename = 'burgers_solution_test01.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'burgers_viscous_time_exact1_test01():' )
  print ( '  Normal end of execution.' )
  return

def r8vec_print ( n, a, title ):

#*****************************************************************************80
#
## r8vec_print() prints an R8VEC.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    31 August 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer N, the dimension of the vector.
#
#    real A(N), the vector to be printed.
#
#    string TITLE, a title.
#
  print ( '' )
  print ( title )
  print ( '' )
  for i in range ( 0, n ):
    print ( '%6d:  %12g' % ( i, a[i] ) )

def hermite_ek_compute ( n ):

#*****************************************************************************80
#
## hermite_ek_compute() computes a Gauss-Hermite quadrature rule.
#
#  Discussion:
#
#    The code uses an algorithm by Elhay and Kautsky.
#
#    The abscissas are the zeros of the N-th order Hermite polynomial.
#
#    The integral:
#
#      integral ( -oo < x < +oo ) exp ( - x * x ) * f(x) dx
#
#    The quadrature rule:
#
#      sum ( 1 <= i <= n ) w(i) * f ( x(i) )
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#  Input:
#
#    integer N, the number of abscissas.
#
#  Output:
#
#    real X(N), the abscissas.
#
#    real W(N), the weights.
#
  from scipy.special import gamma
  import numpy as np
#
#  Define the zero-th moment.
#
  zemu = gamma ( 0.5 )
#
#  Define the Jacobi matrix.
#
  bj = np.zeros ( n )
  for i in range ( 0, n ):
    bj[i] = np.sqrt ( float ( i + 1 ) / 2.0 )

  x = np.zeros ( n )

  w = np.zeros ( n )
  w[0] = np.sqrt ( zemu )
#
#  Diagonalize the Jacobi matrix.
#
  x, w = imtqlx ( n, x, bj, w )
#
#  If N is odd, force the center X to be exactly 0.
#
  if ( ( n % 2 ) == 1 ):
    x[(n-1)//2] = 0.0

  for i in range ( 0, n ):
    w[i] = w[i] ** 2

  return x, w



def imtqlx ( n, d, e, z ):

#*****************************************************************************80
#
## imtqlx() diagonalizes a symmetric tridiagonal matrix.
#
#  Discussion:
#
#    This routine is a slightly modified version of the EISPACK routine to
#    perform the implicit QL algorithm on a symmetric tridiagonal matrix.
#
#    The authors thank the authors of EISPACK for permission to use this
#    routine.
#
#    It has been modified to produce the product Q' * Z, where Z is an input
#    vector and Q is the orthogonal matrix diagonalizing the input matrix.
#    The changes consist (essentially) of applying the orthogonal 
#    transformations directly to Z as they are generated.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#    Roger Martin, James Wilkinson,
#    The Implicit QL Algorithm,
#    Numerische Mathematik,
#    Volume 12, Number 5, December 1968, pages 377-383.
#
#  Input:
#
#    integer N, the order of the matrix.
#
#    real D(N), the diagonal entries of the matrix.
#
#    real E(N), the subdiagonal entries of the
#    matrix, in entries E(1) through E(N-1). 
#
#    real Z(N), a vector to be operated on.
#
#  Output:
#
#    real LAM(N), the diagonal entries of the diagonalized matrix.
#
#    real QTZ(N), the value of Q' * Z, where Q is the matrix that 
#    diagonalizes the input symmetric tridiagonal matrix.
#
  import numpy as np

  lam = np.zeros ( n )
  for i in range ( 0, n ):
    lam[i] = d[i]

  qtz = np.zeros ( n )
  for i in range ( 0, n ):
    qtz[i] = z[i]

  if ( n == 1 ):
    return lam, qtz

  itn = 30

  epsilon = np.finfo(float).eps

  e[n-1] = 0.0

  for l in range ( 1, n + 1 ):

    j = 0

    while ( True ):

      for m in range ( l, n + 1 ):

        if ( m == n ):
          break

        if ( abs ( e[m-1] ) <= epsilon * ( abs ( lam[m-1] ) + abs ( lam[m] ) ) ):
          break

      p = lam[l-1]

      if ( m == l ):
        break

      if ( itn <= j ):
        print ( '' )
        print ( 'imtqlx - Fatal error!' )
        print ( '  Iteration limit exceeded.' )
        raise Exception ( 'imtqlx - Fatal error!' )

      j = j + 1
      g = ( lam[l] - p ) / ( 2.0 * e[l-1] )
      r = np.sqrt ( g * g + 1.0 )

      if ( g < 0.0 ):
        t = g - r
      else:
        t = g + r

      g = lam[m-1] - p + e[l-1] / ( g + t )
 
      s = 1.0
      c = 1.0
      p = 0.0
      mml = m - l

      for ii in range ( 1, mml + 1 ):

        i = m - ii
        f = s * e[i-1]
        b = c * e[i-1]

        if ( abs ( g ) <= abs ( f ) ):
          c = g / f
          r = np.sqrt ( c * c + 1.0 )
          e[i] = f * r
          s = 1.0 / r
          c = c * s
        else:
          s = f / g
          r = np.sqrt ( s * s + 1.0 )
          e[i] = g * r
          c = 1.0 / r
          s = s * c

        g = lam[i] - p
        r = ( lam[i-1] - g ) * s + 2.0 * c * b
        p = s * r
        lam[i] = g + p
        g = c * r - b
        f = qtz[i]
        qtz[i]   = s * qtz[i-1] + c * f
        qtz[i-1] = c * qtz[i-1] - s * f

      lam[l-1] = lam[l-1] - p
      e[l-1] = g
      e[m-1] = 0.0

  for ii in range ( 2, n + 1 ):

     i = ii - 1
     k = i
     p = lam[i-1]

     for j in range ( ii, n + 1 ):

       if ( lam[j-1] < p ):
         k = j
         p = lam[j-1]

     if ( k != i ):

       lam[k-1] = lam[i-1]
       lam[i-1] = p

       p        = qtz[i-1]
       qtz[i-1] = qtz[k-1]
       qtz[k-1] = p

  return lam, qtz


def r8mat_print ( m, n, a, title ):

#*****************************************************************************80
#
## r8mat_print() prints an R8MAT.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    31 August 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer M, the number of rows in A.
#
#    integer N, the number of columns in A.
#
#    real A(M,N), the matrix.
#
#    string TITLE, a title.
#
  r8mat_print_some ( m, n, a, 0, 0, m - 1, n - 1, title )

  return



def r8mat_print_some ( m, n, a, ilo, jlo, ihi, jhi, title ):

#*****************************************************************************80
#
## r8mat_print_some() prints out a portion of an R8MAT.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    10 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer M, N, the number of rows and columns of the matrix.
#
#    real A(M,N), an M by N matrix to be printed.
#
#    integer ILO, JLO, the first row and column to print.
#
#    integer IHI, JHI, the last row and column to print.
#
#    string TITLE, a title.
#
  incx = 5

  print ( '' )
  print ( title )

  if ( m <= 0 or n <= 0 ):
    print ( '' )
    print ( '  (None)' )
    return

  for j2lo in range ( max ( jlo, 0 ), min ( jhi + 1, n ), incx ):

    j2hi = j2lo + incx - 1
    j2hi = min ( j2hi, n )
    j2hi = min ( j2hi, jhi )
    
    print ( '' )
    print ( '  Col: ', end = '' )

    for j in range ( j2lo, j2hi + 1 ):
      print ( '%7d       ' % ( j ), end = '' )

    print ( '' )
    print ( '  Row' )

    i2lo = max ( ilo, 0 )
    i2hi = min ( ihi, m )

    for i in range ( i2lo, i2hi + 1 ):

      print ( '%7d :' % ( i ), end = '' )
      
      for j in range ( j2lo, j2hi + 1 ):
        print ( '%12g  ' % ( a[i,j] ), end = '' )

      print ( '' )

  return


def r8mat_write ( filename, m, n, a ):

#*****************************************************************************80
#
## r8mat_write() writes an R8MAT to a file.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 October 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    string FILENAME, the name of the output file.
#
#    integer M, the number of rows in A.
#
#    integer N, the number of columns in A.
#
#    real A(M,N), the matrix.
#
  output = open ( filename, 'w' )

  for i in range ( 0, m ):
    for j in range ( 0, n ):
      s = '  %g' % ( a[i,j] )
      output.write ( s )
    output.write ( '\n' )

  output.close ( )

  return

