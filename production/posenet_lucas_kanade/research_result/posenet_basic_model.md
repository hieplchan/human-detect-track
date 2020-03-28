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
# 4. PARAMETER LOAD:
```
features.conv0.conv.weight torch.Size([16, 3, 3, 3])
features.conv0.conv.bias torch.Size([16])
features.conv1.depthwise.weight torch.Size([16, 1, 3, 3])
features.conv1.depthwise.bias torch.Size([16])
features.conv1.pointwise.weight torch.Size([32, 16, 1, 1])
features.conv1.pointwise.bias torch.Size([32])
features.conv2.depthwise.weight torch.Size([32, 1, 3, 3])
features.conv2.depthwise.bias torch.Size([32])
features.conv2.pointwise.weight torch.Size([64, 32, 1, 1])
features.conv2.pointwise.bias torch.Size([64])
features.conv3.depthwise.weight torch.Size([64, 1, 3, 3])
features.conv3.depthwise.bias torch.Size([64])
features.conv3.pointwise.weight torch.Size([64, 64, 1, 1])
features.conv3.pointwise.bias torch.Size([64])
features.conv4.depthwise.weight torch.Size([64, 1, 3, 3])
features.conv4.depthwise.bias torch.Size([64])
features.conv4.pointwise.weight torch.Size([128, 64, 1, 1])
features.conv4.pointwise.bias torch.Size([128])
features.conv5.depthwise.weight torch.Size([128, 1, 3, 3])
features.conv5.depthwise.bias torch.Size([128])
features.conv5.pointwise.weight torch.Size([128, 128, 1, 1])
features.conv5.pointwise.bias torch.Size([128])
features.conv6.depthwise.weight torch.Size([128, 1, 3, 3])
features.conv6.depthwise.bias torch.Size([128])
features.conv6.pointwise.weight torch.Size([256, 128, 1, 1])
features.conv6.pointwise.bias torch.Size([256])
features.conv7.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv7.depthwise.bias torch.Size([256])
features.conv7.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv7.pointwise.bias torch.Size([256])
features.conv8.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv8.depthwise.bias torch.Size([256])
features.conv8.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv8.pointwise.bias torch.Size([256])
features.conv9.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv9.depthwise.bias torch.Size([256])
features.conv9.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv9.pointwise.bias torch.Size([256])
features.conv10.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv10.depthwise.bias torch.Size([256])
features.conv10.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv10.pointwise.bias torch.Size([256])
features.conv11.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv11.depthwise.bias torch.Size([256])
features.conv11.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv11.pointwise.bias torch.Size([256])
features.conv12.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv12.depthwise.bias torch.Size([256])
features.conv12.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv12.pointwise.bias torch.Size([256])
features.conv13.depthwise.weight torch.Size([256, 1, 3, 3])
features.conv13.depthwise.bias torch.Size([256])
features.conv13.pointwise.weight torch.Size([256, 256, 1, 1])
features.conv13.pointwise.bias torch.Size([256])
heatmap.weight torch.Size([17, 256, 1, 1])
heatmap.bias torch.Size([17])
offset.weight torch.Size([34, 256, 1, 1])
offset.bias torch.Size([34])
displacement_fwd.weight torch.Size([32, 256, 1, 1])
displacement_fwd.bias torch.Size([32])
displacement_bwd.weight torch.Size([32, 256, 1, 1])
displacement_bwd.bias torch.Size([32])
```
Total params: 577,459
Trainable params: 577,459
Non-trainable params: 0
