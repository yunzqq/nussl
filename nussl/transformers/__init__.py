"""
                          /[-])//  ___
                     __ --\ `_/~--|  / \
                   /_-/~~--~~ /~~~\\_\ /\
                   |  |___|===|_-- | \ \ \
 _/~~~~~~~~|~~\,   ---|---\___/----|  \/\-\
 ~\________|__/   / // \__ |  ||  / | |   | |
          ,~-|~~~~~\--, | \|--|/~|||  |   | |
          [3-|____---~~ _--'==;/ _,   |   |_|
                      /   /\__|_/  \  \__/--/
                     /---/_\  -___/ |  /,--|
                     /  /\/~--|   | |  \///
                    /  / |-__ \    |/
                   |--/ /      |-- | \
                  \^~~\\/\      \   \/- _
                   \    |  \     |~~\~~| \
                    \    \  \     \   \  | \
                      \    \ |     \   \    \
                       |~~|\/\|     \   \   |
                      |   |/         \_--_- |\
                      |  /            /   |/\/
                       ~~             /  /
                                     |__/
"""

from .. import constants
from randomized_svd import RandomizedSVD
if constants.USE_GPU:
    from nmf_tensorflow import NMF
    from basic_autoencoder import BasicAutoEncoder
    from context_deep_autoencoder import ContextDeepAutoEncoder
    from recurrent_autoencoder import RecurrentAutoEncoder

else:
    from sklearn.decomposition import NMF