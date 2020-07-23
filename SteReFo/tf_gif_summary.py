"""
Allows to add gifs to the tf summarys.
Shamelessly stolen from https://colab.research.google.com/drive/1vgD2HML7Cea_z5c3kPBcsHUIxaEVDiIc#scrollTo=joecr_y70phX&forceEdit=true&offline=true&sandboxMode=true
and slightly modified.

Note: you need to have ffmpeg installed
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_op_util

def _encode_gif(images, fps):
  """Encodes numpy images into gif string.
  Args:
    images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
      `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation
  Returns:
    The encoded gif string.
  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  from subprocess import Popen, PIPE
  h, w, c = images[0].shape
  cmd = [
      'ffmpeg', '-y',
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',
      '-r', '%.02f' % fps,
      '-s', '%dx%d' % (w, h),
      '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
      '-i', '-',
      '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
      '-r', '%.02f' % fps,
      '-f', 'gif',
      '-']
  proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in images:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    err = '\n'.join([' '.join(cmd), err.decode('utf8')])
    raise IOError(err)
  del proc
  return out


def _py_gif_summary(tag, images, max_outputs, fps):
  """Outputs a `Summary` protocol buffer with gif animations.
  Args:
    tag: Name of the summary.
    images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
  Returns:
    The serialized `Summary` protocol buffer.
  Raises:
    ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
  """
  is_bytes = isinstance(tag, bytes)
  if is_bytes:
    tag = tag.decode("utf-8")
  images = np.asarray(images)
  if images.dtype != np.uint8:
    raise ValueError("Tensor must have dtype uint8 for gif summary.")
  if images.ndim != 5:
    raise ValueError("Tensor must be 5-D for gif summary.")
  batch_size, _, height, width, channels = images.shape
  if channels not in (1, 3):
    raise ValueError("Tensors must have 1 or 3 channels for gif summary.")

  summ = tf.Summary()
  num_outputs = min(batch_size, max_outputs)
  for i in range(num_outputs):
    image_summ = tf.Summary.Image()
    image_summ.height = height
    image_summ.width = width
    image_summ.colorspace = channels  # 1: grayscale, 3: RGB
    try:
      image_summ.encoded_image_string = _encode_gif(images[i], fps)
    except (IOError, OSError) as e:
      tf.logging.warning(
          "Unable to encode images to a gif string because either ffmpeg is "
          "not installed or ffmpeg returned an error: %s. Falling back to an "
          "image summary of the first frame in the sequence.", e)
      try:
        from PIL import Image  # pylint: disable=g-import-not-at-top
        import io  # pylint: disable=g-import-not-at-top
        with io.BytesIO() as output:
          Image.fromarray(images[i][0]).save(output, "PNG")
          image_summ.encoded_image_string = output.getvalue()
      except:
        tf.logging.warning(
            "Gif summaries requires ffmpeg or PIL to be installed: %s", e)
        image_summ.encoded_image_string = "".encode('utf-8') if is_bytes else ""
    if num_outputs == 1:
      summ_tag = "{}/gif".format(tag)
    else:
      summ_tag = "{}/gif/{}".format(tag, i)
    summ.value.add(tag=summ_tag, image=image_summ)
  summ_str = summ.SerializeToString()
  return summ_str


def gif_summary(name, tensor, fps, max_outputs=3, collections=None, family=None):
  """Outputs a `Summary` protocol buffer with gif animations.
  Args:
    name: Name of the summary.
    tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
    collections: Optional list of tf.GraphKeys.  The collections to add the
      summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.
  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  tensor = tf.convert_to_tensor(tensor)
  #if summary_op_util.skip_summary():
  #  return tf.constant("") #Not sure what it does but no skip_summary in tf 1.4
  with summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = tf.py_func(
        _py_gif_summary,
        [tag, tensor, max_outputs, fps],
        tf.string,
        stateful=False,
        name=scope)
    summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
  return val


if __name__ == "__main__":
    print("Lo")
    images_shape = (16, 64, 64, 64, 3)  # batch, time, height, width, channels
    images = np.zeros(images_shape, dtype=np.uint8)
    images[:, np.arange(64), :, np.arange(64), :] = 255
    images = tf.convert_to_tensor(images)
    
    writer = tf.summary.FileWriter('./log')
    
    global_step = tf.train.get_or_create_global_step()
    global_step_op = tf.assign_add(global_step, 1)
    
    summary_op = gif_summary('images', images, max_outputs=3, fps=10)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for _ in range(100):
      summary_val, global_step_val = sess.run([summary_op, global_step_op])
      if global_step_val % 20 == 0:
        writer.add_summary(summary_val, global_step_val)
    writer.flush()