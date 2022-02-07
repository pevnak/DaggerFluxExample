using Mill
using Random
using Zygote
using Zygote

#####
#	This is just some helper funs to create samples
#####

function string_node(l)
	ArrayNode(NGramMatrix([randstring(rand(1:10)) for i in 1:l]))
end

function dense_node(l)
	ArrayNode(randn(Float32, 100, l))
end

function bag_node(l, childnode)
	nb = rand(1:10, l)
	BagNode(childnode(sum(nb)), Mill.length2bags(nb))	
end

function categorical_node(l)
	cats = 1:10
	ArrayNode(Mill.maybehotbatch(rand(cats, l), cats))
end

function child_node(l)
	ProductNode(
		(aa = string_node(l),
		 bb = one_bagnode(l, string_node),
		 cc = dense_node(l),
		 dd = categorical_node(l),
		)
	)
end

function sample_sample(l = 1)
	ProductNode(
		(a = string_node(l),
		 b = one_bagnode(l, string_node),
		 c = one_bagnode(l, child_node),
		 d = one_bagnode(l, n -> bag_node(n, categorical_node)),
		 e = dense_node(l),
		 f = categorical_node(l),
		)
	)
end

######
#	Let's mode to the real deal
######

# create one sample and create the model using the Mill
x = sample_sample()
model = reflectinmodel(x)

# Let's generate 100 samples
x = sample_sample(100)
model(x)  # inference
