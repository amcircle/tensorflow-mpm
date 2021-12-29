import tensorflow as tf
from PIL import Image, ImageCms
import numpy as np

def g(x, kappa):
  return 1/(1+(x/kappa)**2)

def getAb(I, dt, kappa):
  batch_size, n, m, c = I.shape
  b = tf.reshape(I, [batch_size, n*m*c])
  image_pad = tf.constant([[0,0], [1,1], [1,1], [0,0]])
  B = tf.pad(I, image_pad, 'REFLECT')

  top = tf.abs(B[:,:-2,1:-1,:] - B[:,1:-1,1:-1,:])
  left = tf.abs(B[:,1:-1,:-2,:] - B[:,1:-1,1:-1,:])
  right = tf.abs(B[:,1:-1,2:,:] - B[:,1:-1,1:-1,:])
  bottom = tf.abs(B[:,2:,1:-1,:] - B[:,1:-1,1:-1,:])

  tlrb = tf.concat([top, left, right, bottom], axis = -1)

  minQ = tf.reduce_min(tlrb, axis = -1, keepdims = True)
  maxQ = tf.reduce_max(tlrb, axis = -1, keepdims = True)
  
  P = g(maxQ, kappa)/g(minQ, kappa)
  D = g(minQ, kappa) - g(maxQ, kappa)

  Npd = tf.math.minimum(P,D)

  new_kappa = maxQ*tf.sqrt(Npd/(1-Npd))
  #new_kappa = tfp.stats.percentile(new_kappa, 50.0, interpolation='midpoint', axis = [1,2,3])

  top = -dt*g(top, kappa)
  left = -dt*g(left, kappa)
  right = -dt*g(right, kappa)
  bottom = -dt*g(bottom, kappa)
  center = tf.ones_like(top) - top - left - right - bottom

  A = tf.concat([top, left, center, right, bottom], axis = -1)
  A = tf.cast(A, tf.float32)
  b = tf.cast(b, tf.float32)

  return A, b, new_kappa

def multAx(A, x):
  batch_size, n, m, c_for_eq = A.shape
  c = c_for_eq // 5
  x = tf.reshape(x, [batch_size, n, m, c])
  image_pad = tf.constant([[0,0], [1,1], [1,1], [0,0]])
  x = tf.pad(x, image_pad, 'REFLECT')

  top = x[:,:-2,1:-1,:]
  left = x[:,1:-1,:-2,:]
  right = x[:,1:-1,2:,:]
  bottom = x[:,2:,1:-1,:]
  center = x[:,1:-1,1:-1,:]

  x = tf.concat([top, left, center, right, bottom], axis = -1)

  result_mult = tf.reduce_sum(tf.transpose(tf.reshape(A*x, [batch_size, n*m, 5, c]), [0, 1, 3, 2]), axis = -1)

  return tf.cast(tf.reshape(result_mult, [batch_size, n*m*c]), tf.float32)

def dot(x,y):
  return tf.reduce_sum(x*y, -1, keepdims = True)

def cg(A,x,b):
  r = b - multAx(A,x)
  z = r

  prevX = 10^100
  all_x = []

  while tf.norm(r) > 1:
    prevX = x
    az = multAx(A,z)
    dotr = dot(r,r)
    alpha = dotr/dot(az,z)
    x = x + alpha*z
    r = r - alpha*az
    betta = dot(r,r)/dotr
    z = r + betta*z
    #all_x.append(x)

  return x, all_x #, tf.stack(all_x, 1)

def mpm(I, dt, kappa):
  batch_size, n, m, c = I.shape
  #dt = tf.sqrt(n*m)*dt
  A, b, kappa = getAb(I, dt, kappa)
  #print(kappa)
  x, all_x = cg(A,b,b)
  I = tf.reshape(x, [batch_size, n, m, c])

  return I, all_x #tf.reshape(all_x, [batch_size, all_x.shape[1], n, m, c])
