# 1. FEATURES MODEL:
```
Sequential(
  (conv0): InputConv(
    (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  )
  (conv1): SeperableConv(
    (depthwise): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16)
    (pointwise): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv2): SeperableConv(
    (depthwise): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32)
    (pointwise): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv3): SeperableConv(
    (depthwise): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
    (pointwise): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv4): SeperableConv(
    (depthwise): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64)
    (pointwise): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv5): SeperableConv(
    (depthwise): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
    (pointwise): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv6): SeperableConv(
    (depthwise): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
    (pointwise): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv7): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv8): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv9): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv10): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv11): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv12): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (conv13): SeperableConv(
    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

# 2. HEATMAP MODEL:
```
Conv2d(256, 17, kernel_size=(1, 1), stride=(1, 1))
```
# 3. OFFSET MODEL:
```
Conv2d(256, 34, kernel_size=(1, 1), stride=(1, 1))
```
# 4. OFFSET MODEL:
