# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
tf.keras.applications models
"""

from typing import Dict
from typing import Optional

import nucleus7 as nc7
import tensorflow as tf


class TFKerasApplicationsPlugin(nc7.model.ModelPlugin):
    """
    Predefined models from tf.keras.applications

    Parameters
    ----------
    keras_application_model_name
        CNN model name
    model_params
        arguments for keras applications model, theses parameters will
        be passed to keras.applications.{keras_application_model_name} as
        kwargs. Check keras.applications documentation

    Attributes
    ----------
    incoming_keys
        * features : tensor of shape  [batch_size, w, h, num_channels],
          tf.float32

    generated_keys
        * features : features after processing
    """

    incoming_keys = [
        "features",
    ]

    generated_keys = [
        "features",
    ]

    def __init__(self,
                 keras_application_model_name: str,
                 model_params: Optional[dict] = None,
                 dropout=None,
                 load_fname=None,
                 load_var_scope=None,
                 **plugin_kwargs):
        super(TFKerasApplicationsPlugin, self).__init__(**plugin_kwargs)
        self.keras_application_model_name = keras_application_model_name
        self.model_params = model_params
        self._keras_application_network = None
        self._model_class = None
        self._model_name = "keras_application_network"

        if plugin_kwargs.get("dropout") is not None:
            raise ValueError("{} is not used".format(dropout))
        if plugin_kwargs.get("load_fname") is not None:
            raise ValueError("{} is not used".format(load_fname),
                             "'load_fname' is equivalent to 'weights' parameter"
                             "of the keras applications model")
        if plugin_kwargs.get("load_var_scope") is not None:
            raise ValueError("{} is not used".format(load_var_scope))

    def build(self):
        super().build()
        self._model_class = getattr(
            tf.keras.applications, self.keras_application_model_name)
        return self

    def predict(self, features) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent predict method has more generic signature
        if self.is_training:
            model_params = self.model_params
        else:
            model_params = {k: v for k, v in self.model_params.items()}
            model_params["weights"] = None
        model = self.add_keras_layer(
            lambda: self._model_class(**model_params),
            name=self._model_name)
        output = model(features)
        return {"features": output}

    def reset_keras_layers(self):
        if self.keras_layers_with_names.get(self._model_name):
            del self.keras_layers_with_names[self._model_name]
        super().reset_keras_layers()
