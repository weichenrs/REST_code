# REST
Remote sEnsing image Segmentation Toolbox

## Abstract
As a fundamental task in remote sensing imagery (RSI) interpretation, semantic segmentation of RSI
aims to assign a category label to every pixel in the RSI. As well known, RSI is primarily characterized
by its large-size nature. To pursue precise segmentation with one category or fine-grained categories,
semantic segmentation of RSI is imbued with a desire for holistic segmentation of whole-scene RSI
(WRI), which normally has a large-size characteristic. However, due to the memory constraint of the
graphics processing unit (GPU), conventional deep learning methods, which struggle to handle seman-
tic segmentation of WRI, are compelled to adopt suboptimal strategies such as cropping or fusion,
resulting in performance degradation. Here, we introduce the fiRst-ever End-to-end whole-Scene RSI
semantic segmentation archiTecture (REST), which can universally support various kinds of encoders
and decoders in a plug-and-play manner, allowing seamless integration with mainstream deep learn-
ing methods for semantic segmentation, including the emerging foundation models. In REST, to
overcome the memory limitation, we propose a novel spatial parallel interaction mechanism, which
combines the divide-and-conquer strategy with parallel computing to achieve global context percep-
tion via information interaction across GPUs, enabling REST to segment the WRI effectively and
efficiently. Theoretically and experimentally, REST shows sublinear throughput-scalability in han-
dling WRI along with expansion of GPUs. Experimental results demonstrate that REST consistently
outperforms existing cropping-based and fusion-based methods across diverse semantic segmenta-
tion tasks ranging from single-class to multi-class segmentation, from multispectral to hyperspectral
imagery, and from satellite to drone platform. The robustness and versatility of REST are expected to
offer a promising solution for semantic segmentation of WRI, with the potential for further extension
to medical image segmentation.

## Code
Coming soon.
