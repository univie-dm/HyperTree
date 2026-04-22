# Making Big Data Small: Unsupervised Learning of

# Deep Hierarchical Embeddings in Low-Dimensional

# Hyperbolic Space

```
Anonymous Author(s)
Affiliation
Address
email
```
## Abstract

1 Hierarchical clustering is a cornerstone of unsupervised learning, yet it has been a
2 neglected method in modern deep learning. The unique geometry of hyperbolic
3 space offers an ideal setting for deep hierarchical clustering as it allows to em-
4 bed tree-like structures with minimal distortion. However, prior attempts have
5 been hampered by significant limitations, including geometric rigidity, a lack of
6 scalability to large datasets, and imprecise formulations of key operations. This
7 paper introduces a novel deep hyperbolic clustering framework that directly ad-
8 dresses these shortcomings through three key advancements. First, we present
9 a generalized and rectified definition of the hyperbolic lowest common ancestor
10 for both the Poincaré Ball and the Lorentz models of arbitrary curvature. Second,
11 to address the critical issue of scalability, we employ a deep encoder that learns
12 clusters in an exceptionally low-dimensional space compared to state of the art
13 Euclidean methods. This makes our approach highly efficient and feasible for large-
14 scale datasets. Finally, we introduce HoroPCA++, an improved and numerically
15 stable dimensionality reduction technique for more faithful and lower distorted
16 visualizations of the resulting hierarchies.

## 17 1 Introduction

18 Hierarchical relations are inherent to many real-world data sets. However, often only a flat structure
19 given by labels is regarded, even though the global structure and the relation between different
20 classes can be equally as relevant. Especially for large and high-dimensional datasets,learning the
21 inherent hierarchical structure and embedding the data in a low-dimensional embedding space
22 are fundamental for many downstream tasks like clustering, vizualisation, or further analysis. The
23 hyperbolic space has been proven to be very suitable for embedding hierarchical structures or nested
24 classes, as its volume grows exponentially with increasing dimensionality, allowing a lower distortion
25 than Euclidean spaces with only polynomial volume growth.

26 However, learning an embedding that considers these structures in a fully unsupervised setting has
27 been elusive so far: its computation does not scale well and the geometric properties that make
28 hyperbolic space so powerful also make it difficult to work with. Standard approaches designed
29 for Euclidean space fail because they do not account for the principles of non-Euclidean geometry.
30 Fundamental operations, such as finding the Lowest Common Ancestor (LCA) of two points, are not
31 straightforward and require complex calculations involving geodesics and non-linear projections.

32 While first approaches towards this problem exist, they suffer from scalability issues restricting it to
33 datasets of only a few hundred data objects and strong geometric restrictions and imprecisions Chami
34 et al. [2020], or they still rely on external information Monath et al. [2019],?], Chami et al. [2020].

```
Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute.
```

35 In contrast, our approach is fully unsupervised and mathematically well-founded: We re-formulate
36 the definition of the hyperbolic lowest common ancestor (LCA) to be correct and more general,
37 being applicable to both the Poincaré Ball and Hyperboloid model across arbitrary curvatures. To
38 address scalability, we develop a fully unsupervised deep learning method that learns latent similarity
39 representations using a contrastive encoder, enabling our framework to process large-scale datasets
40 efficiently.

41 For this, we employ contrastive learning with two objectives: a contrastive loss to effectively learn
42 relevant features and a hierarchical loss that is based on the LCAs in hyperbolic space, pushing the
43 embedding into a tree-like structure while minimizing Dasgupta’s cost. In contrast to previous meth-
44 ods that also employ variations of Dasgupta’s cost, we developed a formulation that is differentiable
45 and independent of the number of nodes, edge density, and the scale of weights while being bounded.
46 This is the first

47 todo

48 Our experiments show that we consistently outperform other methods regarding dendrogram purity
49 as well as Dasgupta’s cost and that our embedding is a suitable space to find meaningful clusters in
50 data with an inherent hierarchal structure.

51 Our main contributions are:

52 1.We present the first method that successfully embeds large, complex, and high-dimensional
53 data in low-dimensional hyperbolic space fully unsupervisedly while considering inherent
54 hierarchical structures

55 2.We fully automatically learn the semantic hierarchical relations of complex datasets using
56 an incredibly low-dimensional embedding space.

57 3. For this, we propose a geometrically consistent definition of LCA in hyperbolic space.

58 4. In extensive experiments on data that is too large for most deep clustering methods we
59 show that our method surpasses state-of-the-art methods consistently regarding dendrogram
60 purity.

## 61 2 Hyperbolic Geometry

62 In this work we follow standard notations and definitions for hyperbolic geometry [Cannon et al.,
63 1997, Ratcliffe, 2006] and employ two models of hyperbolic space: thePoincaré Balland the
64 Hyperboloid. While both models capture the same underlying hyperbolic geometry, they provide
65 distinct representations and offer unique views. They are isometric (distance-preserving) models and

```
defined as simply-connected Riemannian manifolds(Mdc,gM
```
dc
66 )with constant negative sectional
67 curvature−c, wherec∈R> 0.

68 2.1 Poincaré Ball

```
Thed-dimensionalPoincaré Ball(Pdc,gP
```
dc
69 )is defined as the open ball of radius^1 /√ccentered at

```
the originPdc=
```
### 

```
x∈Rd:||x||^2 <^1 /c
endowed with the Riemannian metricgP
```
```
dc
x = (λcx)^2 gE
```
```
d
```
(^70) x,
whereλcx=^2 / 1 −c||x||^2 denotes the conformal factor andgE
d
(^71) x the Euclidean metric tensor. This
72 conformal relationship to Euclidean space can also be expressed by the family of inner products
⟨u,v⟩x: TxPdc×TxPdc→R,(u,v)7→(λcx)^2 ⟨u,v⟩
x∈Pdcinduced by the metric. Here,TxP
d
(^73) cis
74 the tangent space with base pointx∈Pdcand⟨.,.⟩the standard Euclidean dot product.
75 2.2 Hyperboloid
Thed-dimensionalHyperboloidmodel (Hdc,gH
dc
76 )is defined as the forward sheet of a two-
77 sheeted hyperboloid embedded within the(d+ 1)-dimensional Minkowski spaceR^1 ,dwith
Hdc=

### 

(x 0 ,...,xd)∈Rd+1: ⟨x,x⟩L=−^1 /c, x 0 > 0
78. That is, the ambient space is the real

79 vector spaceRd+1endowed with the Minkowski inner product – an indefinite bilinear form defined
80 as⟨x,y⟩L=−x 0 y 0 +x 1 y 1 +···+xdydforx,y∈Rd+1. The Riemannian metric onHdcis then


```
81 obtained by restricting⟨.,.⟩Lto the tangent spaces of the Hyperboloid, yielding a positive-definite
82 metric as required for a Riemannian manifold.
83 We show our embeddings work for any hyperbolic space model because we tested 2...
```
## 84 3 Related Work

### 85 NEEDS TO INCLUDE: EUCLIDEAN DEEP CLUSTERING HYPERBOLIC METRIC LEARN-

```
86 ING CHAMI / GHHC PAPER ON HIERARCHICAL CLUSTERING PROPERTY TABLE: fully
87 unsupervised scalable low-dimensional application domain method_1 method_
88 Related work brainstorm. - mostly look into hyperbolic embeddings. - secondly, look into hierarchical
89 embeddings in euclidean space
90 - deep clustering: mainly delimitation from existing methods, but reference quite a few. Most of
91 them do only give you a flat clustering/labeling. In contrast, we learn a hyperbolic embedding that
92 can be used for diverse downstream tasks, including hierarchical clustering. if we present that in
93 the experiments, we should also present another use case of the embedding (e.g. outlier detection or
94 dimensionality reduction)
95 - hyperbolic computations, geometry, poincaré ball etc should go to math background where it is
96 needed in the main part
```
```
97 -
```
```
98 3.1 Deep Clustering in Euclidean Space
```
99 Our approach to scalability is inspired by the paradigm of Deep Clustering (DC), which has become
100 central to modern unsupervised learning. The core idea is to jointly optimize a deep neural network
101 for representation learning alongside a clustering-specific loss function. This process encourages the
102 network to learn a "cluster-friendly" latent space in which data points belonging to the same group are
103 drawn closer together. Foundational methods like DEC [Xie et al., 2016], IDEC [Guo et al., 2017],
104 and DCN [Yang et al., 2017] demonstrated the power of this approach in Euclidean space. However,
105 these methods are designed exclusively for Euclidean geometry and cannot be directly applied to
106 hyperbolic space. Our work is the first to integrate a modern deep clustering framework with the
107 geometric principles of hyperbolic hierarchical clustering, thereby solving the critical scalability
108 problem of prior hyperbolic methods. Comparison against work such as:

109 • Euclidean: "SeCu"

110 • Euclidean: "SCAN"

111 • Euclidean: "Other contrastive stuff?"

112 • Hyperbolic: "Poincaré embeddings for learning hierarchical representations"

113 • Hyperbolic: "Learning continuous hierarchies in the Lorentz model of hyperbolic geometry"

114 • Hyperbolic: "Neural embeddings of graphs in hyperbolic space"

115 • Hyperbolic: "Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature
116 Hierarchy"

117 • Hyperbolic: "MHCN: A Hyperbolic Neural Network Model for Multi-view Hierarchical
118 Clustering"

119 • Hyperbolic: "Unsupervised Hyperbolic Metric Learning"

120 • Hyperbolic: "From trees to continuous embeddings and back: Hyperbolic hierarchical
121 clustering"

122 • Hyperbolic: "Gradient-based hierarchical clustering using continuous representations of
123 trees in hyperbolic space"

124 • Hyperbolic: "Hyperbolic representation learning: Revisiting and advancing"

125 • "H2em: Learning Hierarchical Hyperbolic Embeddings for Compositional Zero-Shot Learn-
126 ing" – Compositional Zero-Shot learning:


127 • apparently quite some papers on word/ document embeddings. Hype-HAN: Hyperbolic
128 Hierarchical Attention Network for Semantic Embedding

129 • Application for 3d scenes: OpenHype: Hyperbolic Embeddings for Hierarchical Open-
130 Vocabulary Radiance Fields

131 • "Designing Hierarchies for Optimal Hyperbolic Embedding" (telling people how to design
132 an ontology to use it for learning, very supervised / labeling-heavy): "hierarchies are best
133 suited for hyperbolic embeddings when they are wide, and single inheritance, independent
134 of the hierarchy size and imbalance"

135 • vision/language models: HyperPath: Knowledge-Guided Hyperbolic Semantic Hierarchy
136 Modeling for WSI Analysis

137 • relevant: Low-distortion and GPU-compatible Tree Embeddings in Hyperbolic Space

138 Thomas preliminary revised related work:

139 Yan et al. [2021] focuses on unsupervised hyperbolic metric learning method for flat clusters by using
140 the internal hierarchical organization of the data to supervise their log-ratio loss function to reflect
141 the multi-level similarities. Internally a hierarchy is used but the evaluation is on the flat clustering.

142 Lin et al. [2024] focuses on the application of hierarchical clustering to word-document and single-cell
143 RNA-sequencing data.

144 Chamberlain et al. [2017] focuses on vertex classification and edge prediction in graph-structured
145 data such as social networks.

146 Nickel and Kiela [2017, 2018] focus on learning hierarchical representations of symbolic data.

147 Yang et al. [2023] discusses the foundations and mechanisms for hierarchical clustering (such as
148 hierarchy inference, root alignment, and stretching), but do not perform hierarchical clustering. The
149 focus is on learning representations that make hierarchical tasks possible and more accurate.

150 Khrulkov et al. [2020] focuses on image segmentation and the general goal of hierarchical organiza-
151 tion, i.e. they focus on learning the representations, but not hierarchical clustering itself or perform
152 any experiments in that direction.

153 Lin et al. [2023] focuses on multi-view clustering using a contrastive hyperbolic loss to align different
154 views of the same object to establish a hierarchy.

155 Duan et al. [2025] focuses on the task of generalized category discovery in hyperbolic space by
156 guiding the clustering of novel or unknown categories through labeled subsets of the dataset.

157 Recent methods proposing continuous hyperbolic representations for hierarchical clustering present
158 promising alternatives to traditional agglomerative approaches, yet suffer from severe scalability
159 bottlenecks on large image datasets. Monath et al. [2019] parameterizes the hierarchy by assigning
160 a continuous vector to every node. Forndatapoints, this necessitates maintainingn− 1 internal
161 nodes, yieldingO(n)distinct parameters. While the authors employ mini-batch stochastic gradient
162 descent to mitigate this, managing millions of node embeddings in GPU memory remains infeasible.
163 Alternatively, Chami et al. [2020] overcome this structural bottleneck by optimizing only the leaf
164 embeddings. However, their objective function relies on computing exact pairwise similarities
165 across allninstances. These similarities are evaluated directly on raw images rather than lower-
166 dimensional embeddings. Constructing the similarity matrix alone incursO(pcn^2 )compute and
167 memory complexity (fornimages withppixels andcchannels). Thus, even though the subsequent
168 continuous optimization phase scales linearly, theO(n^2 )prerequisite renders the method intractable
169 for large datasets.


170 Additional (gemini deep research generated) related work that may be of interest to us:

171 • Representation Learning (General Hyperbolic Models) - mathematical foundation for em-
172 bedding data into non-Euclidean spaces and adapting deep learning operations to hyperbolic
173 geometry.

174 – Ganea Hyperbolic neural networks - generalizing foundational operations—such as
175 multinomial logistic regression, feed-forward layers, and recurrent neural networks

176 – Shimzu probably as well here then

177 – Hyperbolic Graph Convolutional Neural Networks (Chami) - application of graph
178 convolutional networks to hyperbolic space

179 • Hyperbolic Image Clustering (Flat/General Image Grouping) - group images into semantic
180 clusters without necessarily constructing a full hierarchical tree

181 – Hyperbolic Busemann Learning with Ideal Prototype - prototype-based image cluster-
182 ing

183 – Clustering in Hyperbolic Balls - definitions for the barycenter (centroid), allowing
184 for the execution of actualk-means clustering and Expectation-Maximization (EM)
185 algorithms natively in the Poincaré ball

186 – Comparing Euclidean and Hyperbolic K-Means for Generalized Category Discovery -
187 image classification and novel category discovery are optimized by keeping embeddings
188 in the Lorentz Hyperboloid model and clustering them directly with hyperbolick-
189 means, rather than converting them back to Euclidean space

190 • Hierarchical Image Clustering - beyond flat clustering, using hyperbolic geometry to explic-
191 itly map out multi-level, nested relationships (from general concepts down to specific visual
192 parts) within visual datasets.

193 – Exploring Hierarchical Information in Hyperbolic Space for Self-supervised Image
194 Hashing - This work applied a hierarchical hyperbolick-means algorithm to explicitly
195 construct adaptive semantic structures from images. It uses hierarchical contrastive
196 learning to generate hash codes that preserve the multi-level relationships of visual
197 data.

198 – Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval - learning
199 paradigm to encode highly complex, multi-level visual hierarchies (e.g., scene→
200 object→part) natively in hyperbolic space using an angle-based asymmetric distance
201 metric, eliminating the need for explicit hierarchical labels during training.

202 – HAPPI: Hyperbolic Hierarchical Part Prototypes for Image Recognition - Extending
203 prototype learning to visual hierarchies, this work captures both localized image traits
204 and broader contextual features. It maps visual prototypes in hyperbolic space to
205 explicitly identify how specific object parts (like a bird’s beak) nest within broader
206 visual categories.

```
Table 1: Comparison of different methods based on key properties.
Method Low Dim. Unsupervised Scalable Application Domain
Method A Yes Yes No Genomic sequence clustering and
alignment.
Method B No Yes Yes Large-scale social network analysis.
Method C Yes No Yes Real-time computer vision and ob-
ject tracking.
Method D No No No High-precision medical imaging di-
agnostics.
```

## 207 4 Our Method - Hierarchical Metric Learning

208 4.1 Model Architecture

209 Our approach to hierarchical metric learning involves optimizing a hybrid Euclidean-hyperbolic neural
210 network architecture designed to learn multiscale representations in a low-dimensional hyperbolic
211 space. The network is composed of established Euclidean backbone feature extractors that capture
212 initial data representations, followed by a Euclidean projection head that gradually reduces the feature
213 dimensionality. To bound gradients during training, we apply the feature scaling proposed by Klein
214 et al. [2025].

215 Next, we embed these rescaled features into a geometry naturally suited for hierarchies by mapping

```
the Euclidean outputs to a hyperbolic space(Mdc,gM
```
dc
216 )of even lower dimensionality. To achieve
217 this, the Euclidean features are treated as tangent vectors at the manifold’s origin 0 ̄, and we apply
218 the exponential mapexp ̄ 0 :T ̄ 0 Mdc→ Mdcto project them onto the hyperbolic manifold. Finally,
219 the resulting points are passed through a hyperbolic linear layer. We employ the hyperbolic linear
220 layers proposed by Shimizu et al. [2021] for the Poincaré Ball and van der Klis et al. [2026] for the
221 Hyperboloid. Full details on these advanced layers are provided in Appendix A.2.

```
Figure 1: Model Architecture
```
222 4.2 Loss Function

223 Our model is trained end-to-end by jointly optimizing two objectives. The total loss function combines
224 a contrastive lossLCwith our novel hierarchical lossLH:

```
Ltotal=λCLC+λHLH,
```
225 whereλC, λH∈R> 0 are balancing hyperparameters. To ensures the model learns high-quality initial
226 latent representation, we first pretrain it exclusively on the contrastive objective, before introducing
227 the hierarchical loss to progressively impose multiscale hierarchical structures on the embedding
228 space.

229 4.2.1 Contrastive Loss

230 For the contrastive loss we adopt the InfoNCE loss [Oord et al., 2018], which maximizes the
231 agreement between differently augmented views of the same image (positive pairs), while pushing
232 apart all other images in a batch of sizeB(negative samples), thereby encouraging semantically
233 similar inputs to be grouped together in the latent space:

### LC=

### 1

### 2 B

### X^2 B

```
k=
```
```
−log
```
```
exp(sim(zk,zp(k))/τ)
P 2 B
l=1^1 [l̸=k]exp(sim(zk,zl)/τ)
```
### !

234 Here,zkdenotes the hyperbolic embedding of thek-th image, andp(k)its augmented counterpart.
235 The similarity function is defined as the negative geodesic (shortest path) distance in the hyperbolic
236 manifoldsim(.,.) =−dMdc(.,.). Finally,τ > 0 is a temperature parameter that scales the similarities
237 to control the concentration of the distribution.

238 After pretraining,λHis linearly annealed from zero, gradually shifting the optimization from a purely
239 contrastive objective toward one that also enforces continuous hierarchical structures among the
240 latent representations.


241 4.2.2 Hierarchical Loss

242 To impose a multiscale hierarchical structure on the contrastive embedding space, we introduce a
243 novel hierarchical objectiveLHdesigned to complementLC. Traditionally, the quality of a hierarchy
244 T(binary tree) over a point setZ={z 1 ,...,zn}with pairwise similarities has been measured by
245 discrete objectives such as Dasgupta [2016]’s cost. However, Dasgupta’s cost scales asO(n^3 )for
246 dense graphs and is highly sensitive to the magnitude of edge weights, making it difficult to optimize
247 and to compare across different graph topologies of varying size or density. Wang and Wang [2018]
248 addressed these limitations in the discrete setting by normalizing against the base edge weights,
249 yielding a bounded ratio that accurately reflects hierarchical structures independent of graph size.

250 To leverage these structural priors in neural networks, recent works Chami et al. [2020] proposed
251 a continuous, differentiable relaxation of Dasgupta’s cost using hyperbolic geometry. The core
252 idea is to replace discrete Lowest Common Ancestor (LCA) operations with continuous hyperbolic
253 analogues. While foundational, prior hyperbolic objectives inherit the scaling issues of the original
254 Dasgupta cost.

255 To address this, we introduce our own novel objective functionLHthat adapts Wang and Wang
256 [2018]’s normalized discrete cost into a fully differentiable, continuous hyperbolic objective:

### LH(Z) =

### P

```
t(Nt−N
```
```
min
P t )
t(N
tmax−Ntmin),
```
257 where

```
Nt=wij+wik+wjk−
```
### *

```
wij
wik
wjk
```
### !

```
,στ
```
### 

### 

```
dMdc( ̄ 0 ,LCA(zi,zj))
dMdc( 0 ̄,LCA(zi,zk))
dMdc( ̄ 0 ,LCA(zj,zk))
```
### 

### 

### +

### .

258 Here,Z={z 1 ,...,zn}are the hyperbolic embeddings,LCA(.,.)is the hyperbolic Lowest Com-
259 mon Ancestor,dMdc(.,.)is the geodesic distance, andστis the temperature-scaled softmax function.

260 The pairwise similarity is defined aswij= 1/(1 +dMdc(zi,zj))∈(0,1], ensuring that points closer
261 together in the hyperbolic manifold are assigned disproportionately higher similarity values.

262 Our hierarchical objective functionLHoffers several advantages over prior work. It is differentiable,
263 strictly bounded within[0,1], invariant to the number of nodes, edge density, and weight scaling. This
264 ensures the loss is easily tunable while providing interpretable values that faithfully reflect the true
265 quality of learned hierarchies across diverse datasets. Crucially, this formulation also enables efficient
266 scaling to large datasets as we will later establish a minimum sample complexity for estimatingLH
267 that is independent of the total number of embedded nodesn(Section 4.2.4).

268 In the following, we discuss a key geometric detail necessary for evaluating our hyperbolic loss: the
269 rectification of the hyperbolic LCA formulation.

270 4.2.3 Hyperbolic Lowest Common Ancestors

271 The accuracy of our hierarchical objectiveLHrelies on the precise geometric formulation of the
272 continuous hyperbolic LCA. Many existing works [Monath et al., 2019, Chami et al., 2020, Lin
273 et al., 2023, 2024, Duan et al., 2025] employ the notion of LCAs in hyperbolic space to guide their
274 embeddings. The most intuitive and widely adopted definition characterizes the hyperbolic LCA
275 between two points as the point on their geodesic (shortest path) that is closest to the origin ̄ 0 , typically
276 obtained by orthogonally projecting the origin onto the geodesic connecting the corresponding two
277 points. The latter projection we refer to as the geodesic orthogonal projection.

278 While geometrically intuitive, this standard definition exhibits a critical flaw: it fails to properly
279 handle cases where one point is an ancestor of the other in the implied tree structure. In such instances,
280 the orthogonal projection falls outside the geodesic segment connecting the two points. This violates
281 the fundamental tree property that an LCA must lie strictly between its descendants (Figure 2).

282 To correct this limitation, we propose a rectified definition (Definition 4.1) that ensures the hyperbolic
283 LCA strictly lies on the geodesic segmentγx,y⊂Γx,ybetween pointsxandy. This rectification
284 preserves the essential properties of tree-based LCAs while maintaining geometric consistency.

285 Definition 4.1.Letx,ybe two points in either hyperbolic space modelM, and letπΓx,y( ̄0)denote
286 the geodesic orthogonal projection of the origin 0 ̄onto the geodesicΓx,y. The hyperbolic lowest


```
Figure 2: Illustration of the flaw in the pre-
vious hyperbolic LCA definitions. While
the "wrong LCA point" (orange) lies on the
full geodesic (dashed line), it falls outside
the geodesic segment (solid line) connecting
its two descendants, a problem corrected by
the rectified LCA definition proposed in this
work.
```
287 common ancestor LCA(x,y)is defined as:

```
LCA(x,y) =
```
### 

### 

### 

### 

### 

### 

### 

```
̄ 0 , ifcos(∠x ̄ 0 y) =− 1
arg minz∈{x,y}dMdc( ̄ 0 ,z), ifcos(∠x ̄ 0 y)>cos(∠x ̄ 0 p)or
cos(∠x ̄ 0 y)>cos(∠y ̄ 0 p)or
cos(∠x ̄ 0 y) = 1
πΓx,y( ̄0), otherwise
```
288 The three cases in this definition correspond to distinct geometric configurations. The first handles
289 antipodal points, which need to be handled explicitly due to the non-collinearity assumptions of
290 subsequent Propositions. The second case identifies ancestor-descendant relationships by comparing
291 angular distances, ensuring the LCA is the closest ancestor rather than a point outside the connecting
292 geodesic segment. The third applies the standard orthogonal projection when both points lie on the
293 same side of the origin and the projection falls within their connecting segment.

294 While previous literature has formulated hyperbolic LCA projections exclusively for the Poincaré
295 Ball, we provide the first rigorous formulation for the Hyperboloid model, representing a novel
296 contribution to hyperbolic representation learning. In the following propositions, we formalize the
297 geodesic orthogonal projectionπΓx,y( ̄0)in the respective manifolds. Crucially, these formulas are
298 derived in closed form and generalized to arbitrary curvaturesc, making them highly scalable and
299 adaptable to diverse geometric setups without relying on numerical approximations.

300 Proposition 4.2.Letx,y∈Pdcbe two points in the Poincaré Ball that are not collinear. The geodesic
301 orthogonal projection of the origin 0 ̄onto the geodesicΓx,yconnectingxandyis given by:

```
πΓx,y( ̄0)=
```
### 

###  1 −

```
q
||z||^2 −^1 c
```
```
||z||
```
### 

```
z,
```
302

```
z=
```
### 

```
(1 +c||y||^2 )⟨x,y⟩−(1 +c||x||^2 )||y||^2
```
### 

```
x+
```
### 

```
(1 +c||x||^2 )⟨x,y⟩−(1 +c||y||^2 )||x||^2
```
### 

```
y
2 c(|⟨x,y⟩|^2 −||x||^2 ||y||^2 )
```
### .

303 Proposition 4.3.Letx,y∈Hdcbe two points in the Hyperboloid. The geodesic orthogonal projection
304 of the origin ̄ 0 onto the geodesicΓx,yconnectingxandyis given by:

```
πΓx,y( ̄0)=
```
```
αx+βy
p
−c⟨αx+βy,αx+βy⟩L
```
### ,

305

```
α=
```
### √

```
c(x 0 +c⟨x,y⟩Ly 0 )
1 −(c⟨x,y⟩L)^2
```
```
, β=
```
### √

```
c(y 0 +c⟨x,y⟩Lx 0 )
1 −(c⟨x,y⟩L)^2
```
### .

306 4.2.4 Scalability and Sampling Complexity

307 While our hierarchical loss functionLHprovides a highly robust and geometrically sound measure for
308 enforcing hierarchies, its exact evaluation poses a severe scalability challenge. Computing the exact
309 loss requires iterating over all possible triplets in a dataset of sizen, resulting in a computational
310 complexity ofO(n^3 ). This cubic scaling makes exact computation infeasible for modern, large-scale
311 datasets.


312 To overcome theO(n^3 )bottleneck, we estimate the loss via importance and rejection sampling,
313 and establish a formal sample complexity bound to determine the exact number of sampled triplets
314 required to guarantee a tight approximation with high probability. Letε∈(0,1)denote the maximum
315 acceptable absolute deviation of the estimator from the true loss, andδ∈(0,1)be the maximum
316 allowable probability of exceeding this deviation. Further, letwij= (1 +dMdc(zi,zj))−^1 denote the
317 pairwise similarity between two embedded datapointsziandzj, withwmaxandwminrepresenting
318 the maximum and minimum pairwise similarities across the entire dataset. We establish the minimum
319 number of tripletsm∈Nsuch that

```
P[|LmH(Z)−LH(Z)|≥ε]≤δ,
```
320 whereLmH(Z)is the empirical estimator based onmsampled triplets.

321 BecauseLHis defined as a ratio of sums over the set of all possible triplets, we can restate it in terms
322 of expectations under a uniform distribution:

### LH(Z) =

### P

```
t(Nt−N
```
```
min
P t )
t(N
```
```
max
t −N
```
```
min
t )
```
### =

### E

### 

```
Nt−Ntmin
```
### 

### E

### 

```
Ntmax−Ntmin
```
### .

323 We apply importance sampling to estimate this loss efficiently. The importance sampling approach
324 draws triplets according to the target distributionP(t)∝Ntmax−Ntminand computes the estimator:

```
LmH(Z) =
```
### 1

```
m
```
```
Xm
```
```
l=
```
```
Ntl−Ntminl
Ntmaxl −Ntminl
```
```
, withtl∼P.
```
325 To sample fromP(t)without incurring theO(n^3 )computational cost of evaluating the normalizing
constantZ=

### P

```
t′(N
```
```
max
t′ −N
```
```
min
```
(^326) t′ ), we employ rejection sampling. We define the target distribution
327 P(t)and proposal distributionQ(t)as follows:
P(t) =
Ntmax−Ntmin
Z
, Q(t) =

### 1

### |T|

328 where|T|denotes the total number of triplets in the dataset. To ensure a valid acceptance probability
329 (i.e. ≤ 1 ), we must choose a bounding constantMsuch thatP(t)≤MQ(t)for allt. Since

330 Ntmax−Ntmin≤wmax−wmin(Appendix - Proposition A.2), we setM=|T|(wmaxZ−wmin). The
331 acceptance probability for a sampled triplettis then determined by the ratio of the target density to
332 the scaled proposal density:

```
A(t) =
```
```
P(t)
MQ(t)
```
### =

```
Ntmax−Ntmin
Z
|T|(wmax−wmin)
Z ·
```
```
1
|T|
```
### =

```
Ntmax−Ntmin
wmax−wmin
```
### .

333 Thus, we simply sample a triplettuniformly at random with replacement and accept it with probability
334 A(t). This procedure gracefully bypasses the computational intractability of the normalization
335 constantZentirely.

336 Finally, we establish the minimum sample size required to approximate the true loss to the desired
337 precision and confidence.

338 Theorem 4.4(Minimum Sample Complexity).LetLH(Z)be the true normalized loss andLmH(Z)
339 be the empirical estimator computed usingmtriplets sampled fromP(t)∝Ntmax−Ntmin. For any
340 ε,δ∈(0,1), the sample size bound

```
m≥
```
###  1

### 2 +

```
2
3 ε
```
### 

```
ln
```
###  2

```
δ
```
### 

ε^2
341 guarantees
P[|LmH(Z)−LH(Z)|≥ε]≤δ.

342 This bound is particularly powerful as it guarantees a fixed sample size for a given accuracyεand
343 confidenceδ, ensuring robust performance regardless of the underlying data distribution or the dataset
344 sizen. For example, to achieveε= 0. 01 andδ= 0. 01 , we provably need to sample onlym≈ 27 ′ 000
345 triplets, entirely avoiding theO(n^3 )scaling constraint.


## 346 5 Experiments

347 5.1 Datasets

348 For our experiments we used the following datasets:

349 • CIFAR

350 • CUB200-

351 • Tiny-Imagenet

352 • Imagenet

353 • iNaturalist

354 5.2 Implementation Details

355 Backbone encoder architecture, augmentations for contrastive, ...

356 5.3 Metric Learning

357 Evaluation metric for this experiments, comparison methods + results

358 5.4 Deep Clustering

359 Evaluation metric for this experiments, comparison methods + results with ablation on dimensionality

## 360 6 Conclusion

361 In this work, we introduced a deep hyperbolic hierarchical clustering framework that addresses critical
362 limitations of prior methods. By presenting a rectified and generalized definition of the hyperbolic
363 Lowest Common Ancestor (LCA), employing a scalable autoencoder architecture, and developing the
364 numerically robust HoroPCA++ for visualization, our approach achieves state-of-the-art performance
365 on several benchmark datasets. The framework is more scalable, geometrically precise, and flexible
366 than previous approaches, supporting both the Poincaré Ball and Hyperboloid models with arbitrary
367 curvature.
368 Despite these contributions, our work has limitations. The model was evaluated on standard bench-
369 marks of moderate complexity, and its performance on large-scale, high-resolution datasets with a
370 vast number of classes has not been assessed. The current feed-forward network architecture, while
371 effective, may lack the capacity to capture intricate features in more complex data. Furthermore,
372 while our methods improve numerical robustness, computations in hyperbolic space remain sensitive
373 to floating-point precision. Finally, treating the manifold’s curvature as a fixed hyperparameter
374 imposes a uniform geometric structure on all datasets, which may not be optimal for capturing their
375 unique intrinsic hierarchies.
376 Future work will focus on integrating our hyperbolic objective with more powerful network archi-
377 tectures, such as CNNs and Transformers, to tackle more complex datasets. A key direction is to
378 develop methods that allow the manifold’s curvature to be learned as a model parameter, enabling the
379 geometry to adapt to the data’s intrinsic structure. Scaling these methods to even larger datasets and
380 exploring their application to non-image data, such as text and graphs, remains an exciting avenue for
381 further research.

## 382 References

383 Sergei Bernstein. On a modification of chebyshev’s inequality and of the error formula of laplace.
384 Ann. Sci. Inst. Sav. Ukraine, Sect. Math, 1(4):38–49, 1924.

385 Rajendra Bhatia and Chandler Davis. A better bound on the variance.The american mathematical
386 monthly, 107(4):353–357, 2000.

387 James W Cannon, William J Floyd, Richard Kenyon, Walter R Parry, et al. Hyperbolic geometry.
388 Flavors of geometry, 31(59-115):2, 1997.


389 Benjamin Paul Chamberlain, James Clough, and Marc Peter Deisenroth. Neural embeddings of
390 graphs in hyperbolic space.arXiv preprint arXiv:1705.10359, 2017.

391 Ines Chami, Albert Gu, Vaggos Chatziafratis, and Christopher Ré. From trees to continuous embed-
392 dings and back: Hyperbolic hierarchical clustering.Advances in Neural Information Processing
393 Systems, 33:15065–15076, 2020.

394 Sanjoy Dasgupta. A cost function for similarity-based hierarchical clustering. InProceedings of the
395 forty-eighth annual ACM symposium on Theory of Computing, pages 118–127, 2016.

396 Yu Duan, Feiping Nie, Huimin Chen, Zhanxuan Hu, Rong Wang, and Xuelong Li. Hyperbolic
397 hierarchical representation learning for generalized category discovery.IEEE Transactions on
398 Neural Networks and Learning Systems, 2025.

399 Octavian Ganea, Gary Bécigneul, and Thomas Hofmann. Hyperbolic neural networks.Advances in
400 neural information processing systems, 31, 2018.

401 Xifeng Guo, Long Gao, Xinwang Liu, and Jianping Yin. Improved deep embedded clustering
402 with local structure preservation. InIJCAI, pages 1753–1759. ijcai.org, 2017. URLhttps:
403 //dl.acm.org/doi/abs/10.5555/3172077.3172131.

404 William W Hager. Updating the inverse of a matrix.SIAM review, 31(2):221–239, 1989.

405 Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky.
406 Hyperbolic image embeddings. InProceedings of the IEEE/CVF conference on computer vision
407 and pattern recognition, pages 6418–6428, 2020.

408 Timo Klein, Thomas Lang, Andrii Shkabrii, Alexander Sturm, Kevin Sidak, Lukas Miklautz, Claudia
409 Plant, Yllka Velaj, and Sebastian Tschiatschek. Understanding and improving hyperbolic deep
410 reinforcement learning.arXiv preprint arXiv:2512.14202, 2025.

411 Fangfei Lin, Bing Bai, Yiwen Guo, Hao Chen, Yazhou Ren, and Zenglin Xu. Mhcn: A hyperbolic
412 neural network model for multi-view hierarchical clustering. InProceedings of the IEEE/CVF
413 international conference on computer vision, pages 16525–16535, 2023.

414 Ya-Wei Eileen Lin, Ronald R Coifman, Gal Mishne, and Ronen Talmon. Tree-wasserstein distance
415 for high dimensional data with a latent feature hierarchy.arXiv preprint arXiv:2410.21107, 2024.

416 Nicholas Monath, Manzil Zaheer, Daniel Silva, Andrew McCallum, and Amr Ahmed. Gradient-
417 based hierarchical clustering using continuous representations of trees in hyperbolic space. In
418 Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &
419 Data Mining, KDD ’19, page 714–722, New York, NY, USA, 2019. Association for Computing
420 Machinery. ISBN 9781450362016. doi: 10.1145/3292500.3330997. URLhttps://doi.org/
421 10.1145/3292500.3330997.

422 Maximillian Nickel and Douwe Kiela. Poincaré embeddings for learning hierarchical representations.
423 Advances in neural information processing systems, 30, 2017.

424 Maximillian Nickel and Douwe Kiela. Learning continuous hierarchies in the lorentz model of
425 hyperbolic geometry. InInternational conference on machine learning, pages 3779–3788. PMLR,
426 2018.

427 Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
428 coding.arXiv preprint arXiv:1807.03748, 2018.

429 John G Ratcliffe.Foundations of hyperbolic manifolds. Springer, 2006.

430 Ryohei Shimizu, Yusuke Mukuta, and Tatsuya Harada. Hyperbolic neural networks++. InInterna-
431 tional Conference on Learning Representations (ICLR), 2021.

432 Robert van der Klis, Ricardo Chávez Torres, Max van Spengler, Yuhui Ding, Thomas Hofmann,
433 and Pascal Mettes. Fast and geometrically grounded lorentz neural networks. arXiv preprint
434 arXiv:2601.21529, 2026.


435 Dingkang Wang and Yusu Wang. An improved cost function for hierarchical cluster trees.J. Comput.
436 Geom., 11:283–331, 2018. URLhttps://api.semanticscholar.org/CorpusID:54446891.

437 Junyuan Xie, Ross B. Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering
438 analysis. InICML, volume 48 ofJMLR Workshop and Conference Proceedings, pages 478–487.
439 JMLR.org, 2016. URLhttps://dl.acm.org/doi/10.5555/3045390.3045442.

440 Jiexi Yan, Lei Luo, Cheng Deng, and Heng Huang. Unsupervised hyperbolic metric learning. In
441 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
442 12465–12474, 2021.

443 Bo Yang, Xiao Fu, Nicholas D. Sidiropoulos, and Mingyi Hong. Towards k-means-friendly spaces:
444 Simultaneous deep learning and clustering. InICML, volume 70 ofProceedings of Machine
445 Learning Research, pages 3861–3870. PMLR, 2017. URLhttps://dl.acm.org/doi/10.
446 5555/3305890.3306080.

447 Menglin Yang, Min Zhou, Rex Ying, Yankai Chen, and Irwin King. Hyperbolic representation
448 learning: Revisiting and advancing. InInternational conference on machine learning, pages
449 39639–39659. PMLR, 2023.


## 450 A Technical appendices and supplementary material

451 Table 2 summarizes the contents of our Appendix:

```
Appendix Section Content
```
```
Appendix A.1 Parameter Summary Table
```
```
Appendix A.2 Hyperbolic Space Operations
```
```
Appendix A.3 Theorems & Proofs
```
```
Appendix A.4 Datasets & Evaluation Metrics
```
```
Appendix A.5 Experiment Details
```
```
Appendix A.6 Ablation Studies
```
```
Table 2: Structure of our Appendix.
```
452 A.1 Parameter Summary Table

453 Express this as table:

454 • d... latent space dimension (=clustering space)

455 • n... dataset size

456 • B... batch size

457 • m... minimum triplet sample size

458 • Mdc... general hyperbolic space (either Poincaré BallPdcor HyperboloidHdc)

459 • ̄ 0 ... hyperbolic space origin

460 • TxMdc... tangent space atx∈Mdc

461 • LC... contrastive InfoNCE (=NT-Xent) loss with loss weightλC

462 • LH... hierarchical (hyperbolic) loss with loss weightλH

463 • Nt... hierarchical loss of a single triplett= (i,j,k)∈T

464 • Ntmin... minimum hierarchical loss of a single triplett= (i,j,k)∈T

465 • Ntmax... maximum hierarchical loss of a single triplett= (i,j,k)∈T

466 • ε... maximum absolute deviation from true hierarchical loss

467 • δ... confidence for hierarchical loss approximation

468 • Z... embedding datapoints into the hyperbolic clustering space

469 • Γx,y... geodesic (line) connectingxandyextended indefinitely to infinity

470 • πΓx,y(z)... geodesic orthogonal projection ofzonto the geodesic connectingxandy

471 • expx.. exponential map applied atx∈Mdc

472 • dMdc(x,y)... length of the geodesic segment connectingxandy

473 • wij... pairwise similarity betweenziandzj

474 • LCA(.,.)... hyperbolic Lowest Common Ancestor

475 • στis the temperature-scaled softmax function


476 A.2 Hyperbolic Space Operations

477 A.2.1 Möbius Addition

478 In the Poincaré Ball,Möbius additionx⊕yserves as the hyperbolic analogue to Euclidean vector
479 translationx+y. Since the Poincaré Ball is a bounded Riemannian manifold, standard Euclidean
480 addition is not closed. To preserve the geometric structure and ensure the resulting point remains
481 strictly within the manifold, Möbius addition is defined as:

```
x⊕y=
```
```
(1 + 2c⟨x,y⟩+c∥y∥^2 )x+ (1−c∥x∥^2 )y
1 + 2c⟨x,y⟩+c^2 ∥x∥^2 ∥y∥^2
```
```
∀x,y∈Pdc.
```
482 A.2.2 Distance

483 For any two pointsx,y∈Mdc, thegeodesic distancedMdc(x,y)represents the length of the shortest
484 path connecting them along the hyperbolic manifold.

```
Manifold Geodesic Distance
```
```
Poincaré Ball dMdc(x,y) =
```
### 2

### √

```
c
```
```
artanh
```
### √

```
c∥−x⊕y∥
```
### 

```
Hyperboloid dMdc(x,y) =
```
### 1

### √

```
c
```
```
arcosh(−c⟨x,y⟩L)
```
```
Table 3: Geodesic Distance formulas for the Poincaré Ball and Hyperboloid.
```
485 A.2.3 Exponential Map

486 Theexponential mapexpx:TxMdc→ Mdcat a pointx∈ Mdcmaps tangent vectorsv∈ TxMdc
487 to the manifold such that the curvet∈[0,1]7→expx(tv)is a geodesic joiningx=expx( 0 )
488 andexpx(v). Since geodesics are only defined locally, the exponential map, which relies on their
489 existence, is also only defined locally. However, in geodesically complete manifolds such as the
490 Poincaré Ball or the Hyperboloid, these local definitions can be extended indefinitely, allowing the
491 exponential map to be defined on the entire tangent spaceTxMdc.

```
Manifold Exponential Map
```
```
Poincaré Ball expx(v) =x⊕
```
### 

### 

```
tanh
```
### √

```
cλcx∥v∥
2
```
### 

### √

```
c∥v∥
```
### 

```
v
```
```
Hyperboloid expx(v) = cosh
```
### √

```
c∥v∥L
```
### 

```
x+ sinh
```
### √

```
c∥v∥L
```
```
 v
√
c∥v∥L
```
```
Table 4: Exponential Map formulas for the Poincaré Ball and Hyperboloid.
```
492 A.2.4 Parallel Transport

493 Parallel transportis a linear isometryPTx→y:TxMdc→ TyMdcmapping tangent vectorsv∈
494 TxMdcalong geodesics to other tangent spacesTyMdc. Parallel transport allows us to move the local
495 geometry of manifolds along geodesics and thus defines a way of connecting geometries of nearby
496 points.

497 A.2.5 Feature Scaling

498 Feature scalingis a method to mitigate optimization instabilities in hyperbolic representation learning.
499 It prevents the norms of Euclidean embeddings from growing excessively large, thus effectively


```
Manifold Parallel Transport
```
```
Poincaré Ball PTx→y(v) =
```
```
λcx
λcy
```
```
(−(y⊕−x)⊕(y⊕(−x⊕v)))
```
```
Hyperboloid PTx→y(v) =v+
```
```
⟨v,y⟩L
1
c−⟨x,y⟩L
```
```
(x+y)
```
```
Table 5: Parallel Transport formulas for the Poincaré Ball and Hyperboloid.
```
500 bounding the gradients of the hyperbolic layers while preserving the model’s maximal representational
501 capacity [Klein et al., 2025]. To ensure gradient stability, the Euclidean embedding norms are bounded
502 via a learnable scalarξθ∈R, which scales Euclidean embeddingsx∈Rdas follows:

```
xscaled=σ(ξθ)
```
```
atanh(α)
√
c
```
```
f(RMS(x))
√
d
```
### ,

```
whereσ(.)is the sigmoid function,α∈(0,1), andRMS(x) =x/μ(x)withμ(x) =
```
q
503 ε+^1 d∥x∥^2.

504 In our experiments, we chooseα= 0. 95 andfto be the identity function, as it satisfies the desired
505 properties of being 1-Lipschitz continuous with a fixed point at 0.

506 A.2.6 Hyperbolic Linear Layers

507 In Euclidean space, a standard affine transformation of the inputx∈Rdalong thek-th output
508 coordinate takes the form

```
vk(x) =⟨ak,x⟩−bk=⟨ak,−pk+x⟩, bk=⟨ak,pk⟩
```
509 which can be interpreted as a signed distance fromxto the hyperplane with normalak∈Rdand
510 offsetpk∈Rd:

```
vk(x) =⟨ak,−pk+x⟩=∥ak∥sign(⟨ak,−pk+x⟩)d(x,Hak,pk)
```
511 This signed-distance interpretation admits a natural generalization to hyperbolic geometry. Ganea
512 et al. [2018] introduced this hyperbolic counterpart using the algebraic structure of the Poincaré Ball:

```
̃vk(x) =∥ak∥TpkMdcsign (⟨ak,−pk⊕x⟩)dMdc(x,H ̃ak,pk),
```
513 where⊕denotes the Möbius addition andH ̃ak,pk ={z∈Pdc:⟨ak,−pk⊕z⟩= 0}is the
514 corresponding geodesic hyperplane. While ̃vk(x)was originally formulated using Poincaré Ball
515 specific operations, the underlying geometric principle – computing the distance to a geodesic
516 hyperplane defined by a normal vectorakand offsetpk– applies to any hyperbolic manifoldMdc.

517 The hyperbolic linear layer is constructed by computing the signed geodesic distances ̃vk(x)from
518 the input to a set of learned geodesic hyperplanes and mapping the result back to the manifold as the
519 unique pointy∈Mdcwhose distances to the coordinate hyperplanes through the manifold’s origin
520 match ̃vk(x). For both hyperbolic space models, the learnable parametersZ={zk∈ T ̄ 0 Pdc}Kk=
521 andR={rk∈R}Kk=1are defined in the tangent space at the origin 0 ̄, from which the hyperplane
522 normalak∈TpkMdcand offsetpk∈Mdcare recovered via the exponential map (A.2.3) and parallel
523 transport (A.2.4):

```
ak=PT ̄ 0 →pk(zk), pk=exp ̄ 0
```
### 

```
rk
```
```
zk
∥zk∥
```
### 

### .

524 Poincaré Ball [Ganea et al., 2018]. The outputy∈PKc for inputx∈Pdcis

```
y=
```
```
w
1 +
```
```
p
1 +c∥w∥^2
```
```
, wk=
```
### 1

### √

```
c
```
```
sinh
```
### √

```
c ̃vk(x)
```
### 

### ,

525

```
̃vk(x) =
```
```
2 ∥zk∥
√
c
```
```
sinh−^1
```
### 

```
(1−λcx) sinh(
```
### √

```
crk) +
```
### √

```
cλcxcosh(
```
### √

```
crk)
```
### 

```
zk
∥zk∥
```
```
,x
```
### 

### ,

526 whereλcx= 1 −c^2 ∥x∥ 2 is the conformal factor atx.


527 Hyperboloid [van der Klis et al., 2026]. The outputy∈HKc for inputx∈Hdcis

```
y 0 =
```
```
r
∥y1:K∥^2 +
```
### 1

```
c
```
```
, y1:K=
```
```
sinh (
```
### √

```
c ̃v(x))
√
c
```
### ,

528

```
v ̃k(x) =
```
```
∥zk∥
√
c
```
```
sinh−^1
```
### 

### −

### √

```
cx 0 sinh(
```
### √

```
crk) +
```
### √

```
ccosh(
```
### √

```
crk)
```
### 

```
zk
∥zk∥
```
```
,x1:d
```
### 

### ,

529 wherey 0 andy1:Kdenote the time and space components of the output, respectively. Note that this
530 formulation is a direct adaptation of Ganea et al. [2018] to the Hyperboloid and does not include the
531 activation functions or rescaling of van der Klis et al. [2026].

532 A.3 Theorems & Proofs

533 This chapter of the Appendix is split into two main sections: the proof of the geodesic orthogonal
534 projections (Proposition 4.2 & 4.3) and the proof of the minimum sample complexity (Theorem 4.4).

535 A.3.1 Geodesic Orthogonal Projections

536 Mathematical Preliminaries. Before deriving the explicit formulas for geodesic orthogonal pro-
537 jections in the Poincaré Ball and the Hyperboloid, we briefly recap the form of geodesics in each
538 hyperbolic space:

539 • Poincaré Ball.In the Poincaré Ball, geodesics are arcs of circles that intersect the boundary
540 sphere∂Pdcorthogonally. In the special case where these circles pass through the origin, the
541 geodesics are straight lines through the origin, which occurs when the points are collinear
542 with the origin.

543 • Hyperboloid.In the Hyperboloid, geodesics are intersections ofHdcwith 2 -dimensional
544 linear subspaces of the ambient Minkowski spaceR^1 ,d.

545 Lemma A.1(Sherman-Morrison).LetA∈Rd×dbe an invertible square matrix andu,v∈Rd
546 be column vectors. ThenA+uv⊤is invertible if and only if1 +v⊤A−^1 u̸= 0. In this case, its
547 inverse is given by:

```
(A+uv⊤)−^1 =A−^1 −
```
```
A−^1 uv⊤A−^1
1 +v⊤A−^1 u
```
### .

548 Proof.For the proof, we refer the reader to the original paper by Hager [1989].

549 Geodesic Orthogonal Projections in the Poincaré Ball

550 Proposition 4.2. Letx,y∈Pdcbe two points in the Poincaré Ball that are not collinear. The
551 geodesic orthogonal projection of the origin 0 ̄onto the geodesicΓx,yconnectingxandyis given
552 by:

```
πΓx,y( ̄0)=
```
### 

###  1 −

```
q
∥z∥^2 −^1 c
∥z∥
```
### 

```
z,
```
553

```
z=
```
### 

```
(1 +c∥y∥^2 )⟨x,y⟩−(1 +c∥x∥^2 )∥y∥^2
```
### 

```
x+
```
### 

```
(1 +c∥x∥^2 )⟨x,y⟩−(1 +c∥y∥^2 )∥x∥^2
```
### 

```
y
2 c(|⟨x,y⟩|^2 −∥x∥^2 ∥y∥^2 )
```
### .

554 Proof.By assumption,xandyare non-collinear points on the Poincaré Ball, i.e. the geodesicΓx,y

555 is an arc of a circle that meets the boundaryPdcorthogonally (A.3.1). Using the circle inversion
556 property with respect to the boundary sphere∂Pdc={x∈Rn:∥x∥^2 =^1 /c}, we can compute the
557 inverses ofxandy:

```
x−^1 =
```
```
x
c∥x∥^2
```
```
, y−^1 =
```
```
y
c∥y∥^2
```
### .

558 These points satisfy∥x∥·∥x−^1 ∥=^1 /cand∥y∥·∥y−^1 ∥=^1 /c.


559 1.Finding conditions on the circle center. By construction, the pointsx,y,x−^1 ,y−^1 all lie on
560 the same circleΓx,y. Thus, we can construct two perpendicular-bisecting hyperplanes to derive
561 conditions the centerzhas to satisfy.

562 • First perpendicular bisector condition:

z−

```
x+x−^1
2
```
```
,x−x−^1
```
### 

```
= 0 ⇐⇒ ⟨z,x⟩=
```
```
1 +c∥x∥^2
2 c
```
### . (1)

563 • Second perpendicular bisector condition:

z−

```
y+y−^1
2
```
```
,y−y−^1
```
### 

```
= 0 ⇐⇒ ⟨z,y⟩=
```
```
1 +c∥y∥^2
2 c
```
### . (2)

564 • Span condition:

```
z=αx+βy. (3)
```
565 2.Solving for the coefficients. Combining equations (1) and (3), as well as (2) and (3):

```
⟨αx+βy,x⟩=α∥x∥^2 +β⟨x,y⟩=
```
```
1 +c∥x∥^2
2 c
```
### , (4)

566

```
⟨αx+βy,y⟩=α⟨x,y⟩+β∥y∥^2 =
```
```
1 +c∥y∥^2
2 c
```
### . (5)

567 Expressingαfrom the first combined equation (4):

```
α=
```
```
1 +c∥x∥^2
2 c∥x∥^2
```
### −

```
β⟨x,y⟩
∥x∥^2
```
568 Substitutingαinto the second combined equation (5) determinesβ:

```
β=
```
```
(1 +c∥x∥^2 )⟨x,y⟩−(1 +c∥y∥^2 )∥x∥^2
2 c(|⟨x,y⟩|^2 −∥x∥^2 ∥y∥^2 )
```
### (6)

569 Finally, we solve forα:

```
α=
```
```
(1 +c∥y∥^2 )⟨x,y⟩−(1 +c∥x∥^2 )∥y∥^2
2 c(|⟨x,y⟩|^2 −∥x∥^2 ∥y∥^2 )
```
### . (7)

570 3.Center of the circle. The centerzof the circleΓx,yis then fully determined through its
571 coefficientsα(7) andβ(6):

```
z=
```
### 

```
(1 +c∥y∥^2 )⟨x,y⟩−(1 +c∥x∥^2 )∥y∥^2
```
### 

```
x+
```
### 

```
(1 +c∥x∥^2 )⟨x,y⟩−(1 +c∥y∥^2 )∥x∥^2
```
### 

```
y
2 c(|⟨x,y⟩|^2 −∥x∥^2 ∥y∥^2 )
```
572 4.Geodesic projection. Since the circleΓx,ymeets the boundary orthogonally, we have:

```
1
c
```
```
=∥z∥^2 −r^2 ⇐⇒ r=
```
```
r
∥z∥^2 −
```
### 1

```
c
```
### (8)

573 whereris the radius ofΓx,y. Moreover, we have that the geodesic orthogonal projectionπΓx,y( ̄0)
574 lies on the line from the origin toz, and is the point onΓx,yclosest to the origin. This gives us:

```
πΓx,y( ̄0)= (∥z∥−r)
```
```
z
∥z∥
```
### (9)

575 Combining equations (8) and (9) yields the formula for the geodesic orthogonal projection:

```
πΓx,y( ̄0)= (∥z∥−r)
```
```
z
∥z∥
```
### =

### 

###  1 −

```
q
∥z∥^2 −^1 c
```
```
∥z∥
```
### 

```
z.
```
576


577 Geodesic Orthogonal Projections in the Hyperboloid

578 Proposition 4.3. Letx,y∈Hdcbe two points in the Hyperboloid. The geodesic orthogonal
579 projection of the origin ̄ 0 onto the geodesicΓx,yconnectingxandyis given by:

```
πΓx,y( ̄0)=
```
```
αx+βy
p
−c⟨αx+βy,αx+βy⟩L
```
### ,

580

```
α=
```
### √

```
c(x 0 +c⟨x,y⟩Ly 0 )
1 −(c⟨x,y⟩L)^2
```
```
, β=
```
### √

```
c(y 0 +c⟨x,y⟩Lx 0 )
1 −(c⟨x,y⟩L)^2
```
### .

581 Proof.The geodesic orthogonal projectionπΓx,y( ̄0)is computed in two steps:

582 1. Compute the geodesic orthogonal projectionπspan{x,y}( ̄0)of the origin ̄ 0 onto the linear
583 subspace span{x,y}with respect to the Minkowski inner product⟨.,.⟩L.

584 2. Rescaleπspan{x,y}( ̄0)to be a point on the Hyperboloid:

```
πΓx,y( ̄0)=
```
```
πspan{x,y}( ̄0)
q
−c⟨πspan{x,y}( ̄0),πspan{x,y}( ̄0)⟩L
```
### .

585 LetM= [x|y]∈R(d+1)×^2 denote the matrix with column-vectorsx,y∈Rd+1, ̄ 0 ∈Rd+1the
586 Hyperboloid’s origin written as column-vector,B=diag[− 1 , 1 ,...,1]∈R(d+1)×(d+1)the matrix
587 associated with the Minkowski inner product, andρ=⟨x,y⟩L. The geodesic orthogonal projection
588 πspan{x,y}( ̄0)with respect to⟨.,.⟩Lis given by:

```
πspan{x,y}( ̄0)=M(M⊤BM)−^1 M⊤B ̄ 0.
```
589 SinceM⊤BM ∈R^2 ×^2 is a symmetric matrix with main diagonal entries(⟨x,x⟩L,⟨y,y⟩L) =
590 (−^1 /c,−^1 /c)and off-diagonal elementρwe can apply the Sherman-Morrison Lemma A.1 to directly

591 express the inverse. We setA= (−^1 /c−ρ)I 2 ∈R^2 ×^2 ,u= (ρ,ρ)⊤∈R^2 , andv= (1,1)⊤∈R^2.
592 ThenAis invertible and1 +v⊤A−^1 u=^1 1+−cρcρ̸= 0sincex̸=y. Thus:

```
(M⊤BM)−^1 = (A+uv⊤)−^1 =A−^1 −
```
```
A−^1 uv⊤A−^1
1 +v⊤A−^1 u
```
### =

```
−c
1 −(cρ)^2
```
### 

```
1 cρ
cρ 1
```
### 

593 Next, we computeM⊤B ̄ 0 = (−x^0 /√c,−y^0 /√c)⊤∈R^2 and put everything together:

```
πspan{x,y}( ̄0)=M(M⊤BM)−^1 M⊤B 0 ̄= [x|y]
```
```
−c
1 −(cρ)^2
```
### 

```
1 cρ
cρ 1
```
###  

```
−x^0 /√c
−y^0 /√c
```
### 

### =

594

```
=
```
### √

```
c
1 −(cρ)^2
```
```
((x 0 +cρy 0 )x+ (cρx 0 +y 0 )y).
```
595 Re-substituting forρyields:

```
πspan{x,y}( ̄0)=αx+βy,
```
596 where

```
α=
```
### √

```
c(x 0 +c⟨x,y⟩Ly 0 )
1 −(c⟨x,y⟩L)^2
```
```
, β=
```
### √

```
c(y 0 +c⟨x,y⟩Lx 0 )
1 −(c⟨x,y⟩L)^2
```
597 Finally, rescalingπspan{x,y}( ̄0)such that it is a point on the Hyperboloid yields the geodesic
598 orthogonal projection formula.


599 A.3.2 Minimum Sample Complexity

600 Mathematical Preliminaries. To derive a lower bound on the minimum sample sizemfor our
601 loss function, it is necessary to bound the individual loss components to later apply concentration
602 inequalities. In the following we use the same notation as in the main paper (Section 4.2).

603 Proposition A.2.Lett= (i,j,k)be any triplet of indices,

```
Nt=wij+wik+wjk−
```
### *

```
wij
wik
wjk
```
### !

```
,στ
```
### 

### 

```
dMdc( 0 ̄,LCA(zi,zj))
dMdc( ̄ 0 ,LCA(zi,zk))
dMdc( ̄ 0 ,LCA(zj,zk))
```
### 

### 

### +

### ,

604 and

```
LH(Z) =
```
### P

```
t(Nt−N
```
```
min
P t )
t(N
```
```
max
t −N
```
```
min
t )
```
### .

605 Then the following bounds hold:

```
0 ≤Nt−Ntmin≤Ntmax−Ntmin, 0 ≤LH(Z)≤ 1 ,
```
606 wherewmin= minij(wij)andwmax= maxij(wij), respectively.

607 Proof.Recall that

```
Nt=wij+wik+wjk−pijwij−pikwik−pjkwjk
```
608 wherepis a probability distribution from the softmax function. This can be rewritten as:

```
Nt=pij(wik+wjk) +pik(wij+wjk) +pjk(wij+wik).
```
609 Thus,Ntis a convex combination of the three pairwise sums{wij+wik,wij+wjk,wik+wjk}.
610 By definition,Ntminis the minimum of these three sums andNtmaxis the maximum. Therefore:

```
Ntmin≤Nt≤Ntmax.
```
611 SubtractingNtminfrom all sides yields:

```
0 ≤Nt−Ntmin≤Ntmax−Ntmin. (10)
```
612 Summing inequality (10) over all valid tripletst:

```
0 ≤
```
### X

```
t
```
```
(Nt−Ntmin)≤
```
### X

```
t
```
```
(Ntmax−Ntmin)
```
```
and dividing all three terms by
```
### P

```
t(N
```
```
max
t −N
```
```
min
```
(^613) t )bounds the loss function:

### 0 ≤

### P

```
t(Nt−N
```
```
min
P t )
t(N
tmax−Ntmin)≤
```
### P

```
t(N
```
```
max
t −N
```
```
min
P t )
t(N
tmax−Ntmin)= 1.
```
614

615 Lemma A.3(Bhatia-Davis Inequality). LetYbe a bounded random variable such thata≤Y≤b.
616 Then its varianceσ^2 satisfies:

```
σ^2 ≤(b−E[Y]) (E[Y]−a).
```
617 Proof.For the proof, we refer the reader to the original paper by Bhatia and Davis [2000].

618 Lemma A.4(Bernstein’s Inequality).LetY 1 ,...,Ymbe independent zero-mean random variables
619 with|Yl|≤Mandσ^2 be the variance ofYl. Then for anyτ > 0 :

### P

### "^

### 1

```
m
```
```
Xm
```
```
l=
```
```
Yl
≥τ
```
### #

```
≤2 exp
```
### 

### −

```
mτ^2
2 σ^2 +^23 Mτ
```
### 

### .

620 Proof.For the proof, we refer the reader to the original paper by Bernstein [1924].


621 Minimum Sample Complexity - Theorem & Proof

622 Theorem 4.4(Minimum Sample Complexity). LetLH(Z)be the true normalized loss andLmH(Z)
623 be the empirical estimator computed usingmtriplets sampled fromP(t)∝Ntmax−Ntmin. For any
624 ε,δ∈(0,1), the sample size bound

```
m≥
```
###  1

### 2 +

```
2
3 ε
```
### 

```
ln
```
###  2

```
δ
```
### 

ε^2
625 guarantees
P[|LmH(Z)−LH(Z)|≥ε]≤δ.

626 Proof.We prove the bound using Bernstein’s inequality (Lemma A.4). For our importance sampling
627 approach, we introduce a surrogate variable that directly captures the normalized loss. Specifically,
628 we define

```
Xt=
```
```
Nt−Ntmin
Ntmax−Ntmin
```
```
, witht∼P,
```
629 where any triplet withNtmax= NtminyieldsP(t) = 0and is therefore never sampled. The
630 expectation ofXtunder the target distributionPis given by:

```
Et∼P[Xt] =
```
### X

```
t:P(t)> 0
```
```
P(t)·
```
```
Nt−Ntmin
Ntmax−Ntmin
```
### =

### X

```
t:P(t)> 0
```
```
Ntmax−Ntmin
P
t′(N
```
```
max
t′ −N
```
```
min
t′ )
```
### ·

```
Nt−Ntmin
Ntmax−Ntmin
```
### =

### P

```
t(Nt−N
```
```
min
P t )
t′(N
```
```
max
t′ −N
```
```
min
t′ )
```
### =LH(Z).

```
This confirms that our empirical estimatorLmH(Z) =m^1
```
```
Pm
```
(^631) l=1Xtlis an unbiased estimator of the
632 true loss.
633 From Proposition A.2, we know that 0 ≤Nt−Ntmin≤Ntmax−Ntmin, which constrains the
634 surrogate variable to the range 0 ≤Xt≤ 1. Applying the Bhatia-Davis inequality (Lemma A.3) with
635 boundsa= 0andb= 1provides an upper bound on the variance ofXt:
σX^2 t≤(1−LH(Z))(LH(Z)−0) =LH(Z)(1−LH(Z))≤

### 1

### 4

### .

636 To apply Bernstein’s inequality (Lemma A.4) to the deviation of the mean, we define the centered
637 random variableYt =Xt− LH(Z). By the linearity of expectation,E[Yt] = 0. Since both
638 0 ≤Xt≤ 1 and 0 ≤LH(Z)≤ 1 , we can strictly bound the absolute value of|Yt|:

```
|Yt|=|Xt−LH(Z)|≤max(LH(Z), 1 −LH(Z))≤ 1.
```
639 Thus, we have established the uniform bound|Yt|≤MwithM= 1. Further, the variance ofYtis
640 identical to the variance ofXt, due to shift-invariance, meaningσ^2 Yt≤^14.

641 Finally, we apply Bernstein’s inequality (Lemma A.4) to the sample mean with error thresholdτ=ε:

### P

### "

### 1

```
m
```
```
Xm
```
```
l=
```
```
Ytl
≥ε
```
### #

```
≤2 exp
```
### 

### −

```
mε^2
2 ·^14 +^23 · 1 ·ε
```
### 

```
= 2 exp
```
### 

### −

```
mε^2
1
2 +
```
```
2
3 ε
```
### 

### .

642 Setting the right-hand side equal to the confidence parameterδand solving formyields the sample
643 complexity bound stated in the theorem.

644 A.4 Datasets & Evaluation Metrics

645 A.4.1 Datasets

Short dataset description and statistic + ref to source.
646

647 Datasets:


### 648 • CIFAR100

### 649 • CUB200-2011

650 • Tiny-Imagenet

651 • Imagenet100

652 • iNaturalist

653 Evaluation fairness on CUB-200. Care must be taken when comparing retrieval metrics on CUB-
654 200-2011 against baselines that use ImageNet-pretrained features. ImageNet contains 59 bird-related
655 classes with significant visual overlap with CUB-200 categories, which inflates Recall@1 through
656 implicit label leakage. Moreover, a random linear projection from a supervised ResNet-50 feature
657 space into hyperbolic space already achieves approximately 42% Recall@1, purely due to the
658 Johnson–Lindenstrauss property preserving nearest-neighbor structure. This establishes a non-trivial
659 random baseline that any learned hyperbolic head must surpass. To ensure a fair comparison, we
660 additionally evaluate with features obtained from a self-supervised (SimCLR) encoder pretrained
661 from scratch, which eliminates the label leakage effect.

662 A.4.2 Evaluation Metrics

```
kNN accuracy, Dendrogram purity (versions), and sibling score description, recall@.. + for-
```
663 mulas + ref to source.

664 A.5 Experiment Details

665 A.5.1 Hyperparameters

Adjust old hyperparameters.
666

667 A.6 Ablation Studies

requency of LCA correction (5 seeds, 1 dataset - only clustering needs to be redone with
appropriate metric tracking), ...
668

669 Moved from main paper. Since the learning dynamics of deep hierarchical clustering in hyper-
670 bolic space are not yet well understood, we conducted several ablation studies to characterize our
671 model’s behavior and sensitivity to key design choices. We systematically investigated the following
672 components whose results are reported in the appendix in section??:

673 • Curvature: Our study on the effect of curvature c on the MNIST dataset revealed that model
674 performance is highly sensitive to this hyperparameter. We found that no single curvature
675 value was optimal across all metrics. For instance, a lower curvature (c=0.1) achieved the
676 best Dendrogram Purity with Ward linkage, while a higher curvature (c=1.0) yielded the
677 best performance with complete and average linkage. This finding underscores that the ideal
678 geometry is task-dependent and supports our conclusion that treating curvature as a fixed
679 hyperparameter is a limitation.

680 • Numerical Precision: We also analyzed the impact of floating-point precision by comparing
681 training runs using 64-bit (Float64) and 32-bit (Float32) precision. Our results showed a
682 notable drop in performance for both NMI and ARI when using the lower 32-bit precision.
683 This confirms that the geometric computations in our model are sensitive to numerical
684 precision and that using 64-bit precision is crucial for achieving stable results.

685 Table 7 shows the performance on MNIST across Dendrogram Purity (DP), Spectral NMI, and ARI
686 for different values of curvature c. The following results were obtained on MNIST using the same
687 seed and spectral clustering. In Table 8 we compare the float 64-bit precision to float 32-bit. The
688 performance drop with 32-bit precision suggests that some geometric operations, like orthogonal


```
Parameter Value
General Training
Seeds 728, 395, 291, 18, 887, 93, 108, 480, 868, 134
Batch size 512
Optimizer RiemannianSGD
Precision float64
Pretraining
Epochs 1000
Learning rate 1 × 10 −^3
Weight decay 1 × 10 −^4
Clustering
Epochs 1200
Learning rate 1 × 10 −^2
Weight decay 1 × 10 −^4
LR Scheduler Warmup Cosine (20 epochs)
Architecture
Model Type Feed-forward AE
Hidden Dims [512, 256, 64, 16]
Latent Dim 4 (MNIST/KMNIST), 6 (FMNIST)
Manifold Poincaré Ball
Curvature (c) 0.1
Activation Swish
Hyperbolic Encoding HyperbolicLinearPP(Poincare)/FHCNN(Hyperboloid)
Hyperbolic Decoding Log_EuclideanLinear
LCA Loss
Temperature (τ) 0.01
Subsample Size 150
Table 6: Hyperparameters for all dataset experiments(MNIST, FMNIST, KMNIST, GTSRB).
```
689 projections or distance calculations, accumulate errors that can degrade the quality of the learned
embeddings.

```
Table 7: Ablation study on the effect of curvaturecfor the MNIST dataset. All other hyperparameters
are held constant as described in Table 6. Performance is measured by Dendrogram Purity (DP,
Ward and Complete linkage), Spectral Clustering NMI, and Adjusted Rand Index (ARI). The best
performance for each metric is highlighted inbold.
Curvature (c) DP Ward DP Complete Spectral NMI ARI
10.0 64.6 46.03 68.1 56.9
7.5 55.4 42.6 58.1 43.1
5.0 62.9 50.86 74.2 65.3
2.5 76.7 55.42 78.0 70.2
2.0 73.5 56.27 76.6 69.6
1.0 76.9 65.0 69.82 69.8
0.5 72.8 57.93 74.1 62.9
0.1 81.3 64.6 79.7 71.4
```
690

691 A.6.1 Training Dynamics of the Hyperbolic Head

692 When the contrastive loss is computed exclusively on the Euclidean backbone features—as is the
693 case during pretraining in our framework—the hyperbolic projection head receives no gradient signal
694 during this phase. Consequently, the head enters the clustering stage with effectively random weights.
695 We observed that this leads to a characteristicdip-then-recoverpattern in retrieval metrics (e.g.,


```
Table 8: Comparison of numerical precision on MNIST for spectral clustering. The Float32 results
represent the mean and standard deviation over 3 seeds.
Model Precision NMI Score (%)
```
```
Poincaré Ball
```
```
Float32 74. 22 ± 1. 8
Float64 78. 96 ± 0. 642
```
```
Hyperboloid
```
```
Float32 75. 69 ± 4. 6
Float64 77. 81 ± 0. 025
```
696 Recall@1): performance drops sharply at the onset of hierarchical training before slowly recovering
697 as the head adapts.

698 A natural but incorrect hypothesis is that this dip stems from poor weight initialization. However,
we verified that the initialization scheme (Kaiming scaling with factor 1 /

### √

699 din) is identical across all
700 successful and failing runs, ruling out initialization as the root cause.

701 The actual cause is the absence of any hyperbolic objective during pretraining: unlike earlier con-
702 figurations that employed a dual Euclidean–hyperbolic contrastive loss, the current architecture
703 trains the head only once the hierarchical lossLHis activated. To address this, we introduce ahead
704 warmupphase ofNwarmupepochs at the start of the clustering stage. During warmup,λH= 0and the
705 contrastive loss is temporarily routed through the hyperbolic head via a geodesic NT-Xent objective,
706 giving the head a structured initialization beforeLHramps on.

707 A.6.2 Gradient Absorption in the Hyperbolic Head

708 A related pathology arises when the backbone encoder is fully trainable while the contrastive loss
709 is computed on the hyperbolic projections. In this configuration, the hyperbolic head absorbs the
710 majority of the gradient, and the backbone receives an insufficient learning signal. The head then
711 drifts toward degenerate configurations in which embeddings saturate near the boundary of the
712 Poincaré Ball.

713 We found thatfreezing the encoderis the single most predictive hyperparameter for stable training.
714 Across all configurations tested on CIFAR-100, the top-performing runs consistently used a frozen
715 encoder, while runs with a fully trainable backbone exhibited embedding collapse (Recall@1 dropping
716 from 0.52 to below 0.05). Selectively unfreezing only the final residual block (layer4) with a reduced
717 learning rate factor of 0. 1 preserves the benefits of a frozen backbone while still allowing limited
718 fine-tuning.

719 A.6.3 Hierarchical Loss Annealing

720 Activating the hierarchical lossLHat full weight from the start of the clustering phase can destabilize
721 training: the LCA-based objective pushes embeddings toward the boundary before the contrastive
722 structure has consolidated, leading to degenerate configurations. We therefore initializeλH= 0and
723 linearly anneal it over a warm-up interval of at least 250 epochs. This schedule allows the contrastive
724 and regularization objectives to establish a well-structured embedding before hierarchical constraints
725 are imposed.

726


