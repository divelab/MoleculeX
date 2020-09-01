"""
Configuration file
"""



conf = {}



######################################################################################################################
# Settings for training
##    'epochs': maximum training epochs
##    'early_stopping': patience used to stop training
##    'lr': starting learning rate
##    'lr_decay_factor': learning rate decay factor
##    'lr_decay_step_size': step size of learning rate decay 
##    'dropout': dropout rate
##    'weight_decay': l2 regularizer term
##    'depth': number of layers
##    'batch_size': training batch_size
######################################################################################################################
conf['epochs'] = 400
conf['early_stopping'] = 200
conf['lr'] = 0.0001
conf['lr_decay_factor'] = 0.8
conf['lr_decay_step_size'] = 50
conf['dropout'] = 0
conf['weight_decay'] = 0
conf['depth'] = 3
conf['hidden'] = 256
conf['batch_size'] = 64


######################################################################################################################
# Settings for val/test
##    'vt_batch_size': val/test batch_size
######################################################################################################################
conf['vt_batch_size'] = 1000
