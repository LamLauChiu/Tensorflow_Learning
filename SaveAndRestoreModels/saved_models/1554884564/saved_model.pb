Ź˘
Đ
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
9
Softmax
logits"T
softmax"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*
1.13.0-rc22b'v1.13.0-rc1-19-gc865ec5621'8Î
s
dense_14_inputPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
Ľ
0dense_14/kernel/Initializer/random_uniform/shapeConst*
valueB"     *"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes
:

.dense_14/kernel/Initializer/random_uniform/minConst*
valueB
 *HY˝*"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes
: 

.dense_14/kernel/Initializer/random_uniform/maxConst*
valueB
 *HY=*"
_class
loc:@dense_14/kernel*
dtype0*
_output_shapes
: 
×
8dense_14/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_14/kernel/Initializer/random_uniform/shape*"
_class
loc:@dense_14/kernel*
dtype0* 
_output_shapes
:
*
T0
Ú
.dense_14/kernel/Initializer/random_uniform/subSub.dense_14/kernel/Initializer/random_uniform/max.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel*
_output_shapes
: 
î
.dense_14/kernel/Initializer/random_uniform/mulMul8dense_14/kernel/Initializer/random_uniform/RandomUniform.dense_14/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@dense_14/kernel* 
_output_shapes
:

ŕ
*dense_14/kernel/Initializer/random_uniformAdd.dense_14/kernel/Initializer/random_uniform/mul.dense_14/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_14/kernel* 
_output_shapes
:

 
dense_14/kernelVarHandleOp*
_output_shapes
: *
shape:
* 
shared_namedense_14/kernel*"
_class
loc:@dense_14/kernel*
dtype0
o
0dense_14/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/kernel*
_output_shapes
: 

dense_14/kernel/AssignAssignVariableOpdense_14/kernel*dense_14/kernel/Initializer/random_uniform*"
_class
loc:@dense_14/kernel*
dtype0

#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*"
_class
loc:@dense_14/kernel*
dtype0* 
_output_shapes
:


dense_14/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@dense_14/bias*
dtype0*
_output_shapes	
:

dense_14/biasVarHandleOp*
_output_shapes
: *
shape:*
shared_namedense_14/bias* 
_class
loc:@dense_14/bias*
dtype0
k
.dense_14/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_14/bias*
_output_shapes
: 

dense_14/bias/AssignAssignVariableOpdense_14/biasdense_14/bias/Initializer/zeros* 
_class
loc:@dense_14/bias*
dtype0

!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias* 
_class
loc:@dense_14/bias*
dtype0*
_output_shapes	
:
p
dense_14/MatMul/ReadVariableOpReadVariableOpdense_14/kernel*
dtype0* 
_output_shapes
:

|
dense_14/MatMulMatMuldense_14_inputdense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
dense_14/BiasAdd/ReadVariableOpReadVariableOpdense_14/bias*
dtype0*
_output_shapes	
:

dense_14/BiasAddBiasAdddense_14/MatMuldense_14/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
dense_14/ReluReludense_14/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
dropout_7/IdentityIdentitydense_14/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0dense_15/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes
:

.dense_15/kernel/Initializer/random_uniform/minConst*
valueB
 *Ű˝*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes
: 

.dense_15/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ű=*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes
: 
Ö
8dense_15/kernel/Initializer/random_uniform/RandomUniformRandomUniform0dense_15/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes
:	

Ú
.dense_15/kernel/Initializer/random_uniform/subSub.dense_15/kernel/Initializer/random_uniform/max.dense_15/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes
: 
í
.dense_15/kernel/Initializer/random_uniform/mulMul8dense_15/kernel/Initializer/random_uniform/RandomUniform.dense_15/kernel/Initializer/random_uniform/sub*
_output_shapes
:	
*
T0*"
_class
loc:@dense_15/kernel
ß
*dense_15/kernel/Initializer/random_uniformAdd.dense_15/kernel/Initializer/random_uniform/mul.dense_15/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@dense_15/kernel*
_output_shapes
:	


dense_15/kernelVarHandleOp*
shape:	
* 
shared_namedense_15/kernel*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes
: 
o
0dense_15/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_15/kernel*
_output_shapes
: 

dense_15/kernel/AssignAssignVariableOpdense_15/kernel*dense_15/kernel/Initializer/random_uniform*"
_class
loc:@dense_15/kernel*
dtype0

#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*"
_class
loc:@dense_15/kernel*
dtype0*
_output_shapes
:	


dense_15/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
valueB
*    * 
_class
loc:@dense_15/bias

dense_15/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*
shared_namedense_15/bias* 
_class
loc:@dense_15/bias
k
.dense_15/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_15/bias*
_output_shapes
: 

dense_15/bias/AssignAssignVariableOpdense_15/biasdense_15/bias/Initializer/zeros* 
_class
loc:@dense_15/bias*
dtype0

!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias* 
_class
loc:@dense_15/bias*
dtype0*
_output_shapes
:

o
dense_15/MatMul/ReadVariableOpReadVariableOpdense_15/kernel*
dtype0*
_output_shapes
:	


dense_15/MatMulMatMuldropout_7/Identitydense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

i
dense_15/BiasAdd/ReadVariableOpReadVariableOpdense_15/bias*
dtype0*
_output_shapes
:


dense_15/BiasAddBiasAdddense_15/MatMuldense_15/BiasAdd/ReadVariableOp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
_
dense_15/SoftmaxSoftmaxdense_15/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
-
predict/group_depsNoOp^dense_15/Softmax
U
ConstConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_3Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
\
Const_5Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_7Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_10Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 

RestoreV2/tensor_namesConst*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
c
RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
~
	RestoreV2	RestoreV2Const_5RestoreV2/tensor_namesRestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
L
AssignVariableOpAssignVariableOpdense_14/kernelIdentity*
dtype0

RestoreV2_1/tensor_namesConst*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2Const_5RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
_output_shapes
:*
T0
N
AssignVariableOp_1AssignVariableOpdense_14/bias
Identity_1*
dtype0

RestoreV2_2/tensor_namesConst*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2Const_5RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_2IdentityRestoreV2_2*
_output_shapes
:*
T0
P
AssignVariableOp_2AssignVariableOpdense_15/kernel
Identity_2*
dtype0

RestoreV2_3/tensor_namesConst*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_3	RestoreV2Const_5RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
N
AssignVariableOp_3AssignVariableOpdense_15/bias
Identity_3*
dtype0
Q
VarIsInitializedOpVarIsInitializedOpdense_15/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_1VarIsInitializedOpdense_15/bias*
_output_shapes
: 
Q
VarIsInitializedOp_2VarIsInitializedOpdense_14/bias*
_output_shapes
: 
S
VarIsInitializedOp_3VarIsInitializedOpdense_14/kernel*
_output_shapes
: 
l
initNoOp^dense_14/bias/Assign^dense_14/kernel/Assign^dense_15/bias/Assign^dense_15/kernel/Assign
X
Const_11Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_12Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
Á
SaveV2/tensor_namesConst"/device:CPU:0*ę
valueŕBÝ
B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:


SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B 
š
SaveV2SaveV2Const_12SaveV2/tensor_namesSaveV2/shape_and_slicesConst_6Const_7Const_8Const_9Const_10#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpConst_11"/device:CPU:0*
dtypes
2

Y

Identity_4IdentityConst_12^SaveV2"/device:CPU:0*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:	*Ě
valueÂBż	B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
u
save/SaveV2/shape_and_slicesConst*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
ý	
save/SaveV2/tensors_0Const*
_output_shapes
: *ˇ	
value­	BŞ	 BŁ	{"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_14", "trainable": true, "units": 512, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_7", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}, {"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_15", "trainable": true, "units": 10, "use_bias": true}}], "name": "sequential_7"}}*
dtype0
ă
save/SaveV2/tensors_1Const*
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 784], "dtype": "float32", "name": "dense_14_input", "sparse": false}}*
dtype0*
_output_shapes
: 
ë
save/SaveV2/tensors_2Const*Ľ
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_7", "noise_shape": null, "rate": 0.2, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
Ę
save/SaveV2/tensors_3Const*
dtype0*
_output_shapes
: *
valueúB÷ Bđ{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 784], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_14", "trainable": true, "units": 512, "use_bias": true}}
Ş
save/SaveV2/tensors_6Const*ä
valueÚB× BĐ{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_15", "trainable": true, "units": 10, "use_bias": true}}*
dtype0*
_output_shapes
: 
ő
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/SaveV2/tensors_0save/SaveV2/tensors_1save/SaveV2/tensors_2save/SaveV2/tensors_3!dense_14/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOpsave/SaveV2/tensors_6!dense_15/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Ť
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ě
valueÂBż	B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:	

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Ç
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*8
_output_shapes&
$:::::::::

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp

save/NoOp_3NoOp
N
save/IdentityIdentitysave/RestoreV2:4*
_output_shapes
:*
T0
T
save/AssignVariableOpAssignVariableOpdense_14/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:5*
_output_shapes
:*
T0
Z
save/AssignVariableOp_1AssignVariableOpdense_14/kernelsave/Identity_1*
dtype0

save/NoOp_4NoOp
P
save/Identity_2Identitysave/RestoreV2:7*
_output_shapes
:*
T0
X
save/AssignVariableOp_2AssignVariableOpdense_15/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpdense_15/kernelsave/Identity_3*
dtype0
Â
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3
^save/NoOp^save/NoOp_1^save/NoOp_2^save/NoOp_3^save/NoOp_4

init_1NoOp"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"
	variablesűř

dense_14/kernel:0dense_14/kernel/Assign%dense_14/kernel/Read/ReadVariableOp:0(2,dense_14/kernel/Initializer/random_uniform:08
s
dense_14/bias:0dense_14/bias/Assign#dense_14/bias/Read/ReadVariableOp:0(2!dense_14/bias/Initializer/zeros:08

dense_15/kernel:0dense_15/kernel/Assign%dense_15/kernel/Read/ReadVariableOp:0(2,dense_15/kernel/Initializer/random_uniform:08
s
dense_15/bias:0dense_15/bias/Assign#dense_15/bias/Read/ReadVariableOp:0(2!dense_15/bias/Initializer/zeros:08"
trainable_variablesűř

dense_14/kernel:0dense_14/kernel/Assign%dense_14/kernel/Read/ReadVariableOp:0(2,dense_14/kernel/Initializer/random_uniform:08
s
dense_14/bias:0dense_14/bias/Assign#dense_14/bias/Read/ReadVariableOp:0(2!dense_14/bias/Initializer/zeros:08

dense_15/kernel:0dense_15/kernel/Assign%dense_15/kernel/Read/ReadVariableOp:0(2,dense_15/kernel/Initializer/random_uniform:08
s
dense_15/bias:0dense_15/bias/Assign#dense_15/bias/Read/ReadVariableOp:0(2!dense_15/bias/Initializer/zeros:08*Ł
serving_default
:
dense_14_input(
dense_14_input:0˙˙˙˙˙˙˙˙˙5
dense_15)
dense_15/Softmax:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1