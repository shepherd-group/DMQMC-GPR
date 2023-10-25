#!/usr/bin/env python

import os
import GPy
import numpy

from typing import TypeVar, Any

Array = TypeVar('numpy.ndarray')


class GPyDMQMC:
    ''' A wrapper class for calling GPy on DMQMC data.

    Attributes
    ----------
    x : :class:`numpy.ndarray`
        Contains the x-axis values which are used in generating the
        predicted y-axis values for the trained model.
    xtrain : :class:`numpy.ndarray`
        The x-axis values used to train the model.
    ytrain : :class:`numpy.ndarray`
        The y-axis values used to train the model.
    xpredict : :class:`numpy.ndarray`
        The x values transformed to be a column vector.
    etrain : :class:`numpy.ndarray`, default = None
        The error associated with the ytrain data.
    max_f_eval : int, default = 1000
        Controls the maximum number of optimization steps used by GPy.
    save_file : str, default = ''
        A save file to store the model parameters in, if the file is blank
        no parameters are saved.
    restart_file : str, default = ''
        A file containing model parameters which are read in and used to
        generate the model rather than training on the provided data.
    verbose : bool, default = True
        Print information as training occurs.
    kernel_product : bool, default = False
        Indicates if a pre-defined kernel product should be used for the
        GPR model.
    heteroscedastic : bool, default = False
        Indicates if a heteroscedastic GPR model should be trained, requires
        etrain to be used.
    save_restart : bool
        Set in the _check_options method, will be True when a restart file
        will be saved for the trained model parameters. Otherwise False.
    restarting : bool
        Set in the _check_options method, will be True when a restart file
        will be read in instead of training a model with the training data.
    model : :class:`GPy.models.gp_regression.GPRegression`
        The GPR model from GPy, by default it is homoscedastic but can also
        relate to the heteroscedastic class, e.g. see:
            >> help(GPy.models.GPHeteroscedasticRegression)
        for more information.
    y : :class:`numpy.ndarray`
        The y-axis prediction generated from the GPR model for x-axis values
        contained in x.
    var : :class:`numpy.ndarray`
        The variance associated with the y-axis predictions.
    dy : :class:`numpy.ndarray`
        The predicted values for the derivative of the model, e.g. corresponds
        to f'(x) if we assume the model predicts f(x).
    dvar : :class:`numpy.ndarray`
        The variance associated with the dy predictions.

    Methods
    -------
    _check_options()
        Checks the consistency of input options and sets some additional
        variables based on the input options. Will automatically be called
        on __init__.
    _optimize()
        Run the model optimization, or generate the model from a restart
        file. Will automatically be called on __init__.
    _predict()
        Generate the predictions from the model. Will automatically be called
        on __init__.
    _get_kernel()
        A private and static method which the _optimize() method calls to
        get the kernel used in GPR.
    '''
    def __init__(
                self,
                xtrain: Array,
                ytrain: Array,
                xpredict: Array,
                etrain: Array = None,
                max_f_eval: int = 1000,
                save_file: str = '',
                restart_file: str = '',
                verbose: bool = True,
                kernel_product: bool = False,
                heteroscedastic: bool = False,
            ) -> None:
        ''' Perform GPR on a given data set, and generate the predictions
        for that data set.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Contains the x-axis values which are used in generating the
            predicted y-axis values for the trained model.
        xtrain : :class:`numpy.ndarray`
            The x-axis values used to train the model.
        ytrain : :class:`numpy.ndarray`
            The y-axis values used to train the model.
        xpredict : :class:`numpy.ndarray`
            The x values transformed to be a column vector.
        etrain : :class:`numpy.ndarray`, default = None
            The error associated with the ytrain data.
        max_f_eval : int, default = 1000
            Controls the maximum number of optimization steps used by GPy.
        save_file : str, default = ''
            A save file to store the model parameters in, if the file is blank
            no parameters are saved.
        restart_file : str, default = ''
            A file containing model parameters which are read in and used to
            generate the model rather than training on the provided data.
        verbose : bool, default = True
            Print information as training occurs.
        kernel_product : bool, default = False
            Indicates if a pre-defined kernel product should be used for the
            GPR model.
        heteroscedastic : bool, default = False
            Indicates if a heteroscedastic GPR model should be trained,
            requires etrain to be used.

        Returns
        -------
        None.
        '''
        self.x = xpredict
        self.xtrain = xtrain[:, None]
        self.ytrain = ytrain[:, None]
        self.xpredict = xpredict.reshape(len(xpredict), 1)
        self.etrain = etrain
        self.max_f_eval = max_f_eval
        self.save_file = save_file
        self.restart_file = restart_file
        self.verbose = verbose
        self.kernel_product = kernel_product
        self.heteroscedastic = heteroscedastic

        self._check_options()
        self._optimize()
        self._predict()

    def _check_options(self) -> None:
        ''' Checks that the input parameters are not in conflict and
        set some variables for performing GPR.
        '''
        if self.heteroscedastic and self.etrain is None:
            raise RuntimeError('Must provide the error associated with the '
                               'data set using the etrain parameter to use '
                               'the heteroscedastic model!')

        if self.save_file != '' and os.path.isfile(self.save_file):
            raise RuntimeError('The provided save file:\n'
                               f'    {self.save_file}\n'
                               'already exist, please use a unique name!')
        elif self.save_file != '':
            self.save_restart = True
        else:
            self.save_restart = False

        if self.restart_file != '' and not os.path.isfile(self.restart_file):
            raise RuntimeError('The provided restart file:\n'
                               f'    {self.restart_file}\ndoes not exist!')
        elif self.restart_file != '':
            self.restarting = True
        else:
            self.restarting = False

    def _optimize(self) -> None:
        ''' Do the model optimization. Once complete a model will be
        able to generate predictions for the x-axis prediction values
        provided by the user.
        '''
        kernel = self._get_kernel(self.kernel_product)

        if self.heteroscedastic:
            self.model = GPy.models.GPHeteroscedasticRegression(
                    self.xtrain,
                    self.ytrain,
                    kernel,
                    initialize=not self.restarting,
                )
            self.etrain = numpy.abs(self.etrain)[:, None]
            self.model['.*het_Gauss.variance'] = self.etrain
            self.model.het_Gauss.variance.fix()
        else:
            self.model = GPy.models.GPRegression(
                    self.xtrain,
                    self.ytrain,
                    kernel,
                    initialize=not self.restarting,
                )

        if self.restarting:
            self.model.update_model(False)
            self.model.initialize_parameter()
            self.model[:] = numpy.load(self.restart_file)
            self.model.update_model(True)
        else:
            if self.verbose:
                print('\n -- Pre fitting -- ')
                print(self.model)

            self.model.optimize(
                    messages=self.verbose,
                    max_f_eval=self.max_f_eval,
                )

        if self.verbose:
            print('\n -- Post fitting -- ')
            print(self.model)

        if self.save_restart:
            print(f'\n -- Saving restart to: {self.save_file} --')
            numpy.save(self.save_file, self.model.param_array)
            print()

    def _predict(self) -> None:
        ''' Generate the predictions from the trained model.
        The predictions are stored in the attributes `y`, `var`, `dy`, and
        `dvar`. For more information see the attributes for the class.
        '''
        if self.heteroscedastic:
            prediction = self.model._raw_predict(self.xpredict)
        else:
            prediction = self.model.predict(self.xpredict)

        self.y, self.var = [k.flatten() for k in prediction]

        gradient = self.model.predictive_gradients(self.xpredict)

        self.dy, _ = [k.flatten() for k in gradient]

    @staticmethod
    def _get_kernel(kernel_product: bool) -> Any:
        ''' A private method which forms and returns the kernel.

        Parameters
        ----------
        kernel_product : bool
            True if we are using the kernel product as outlined in:
            https://doi.org/10.1063/5.0150702.

        Returns
        -------
        kernel : :class:`GPy.kern.src.add.Add`
            Contains the kernel to be used in optimization for GPR.
        '''
        if kernel_product:
            kernel = GPy.kern.RBF(
                    1,
                    ARD=False,
                ) + GPy.kern.Matern52(
                    1,
                    ARD=False,
                ) + GPy.kern.Matern32(
                    1,
                    ARD=False,
                ) + GPy.kern.RBF(
                    1,
                    ARD=False,
                )*GPy.kern.Matern52(
                    1,
                    ARD=False,
                )
        else:
            kernel = GPy.kern.RBF(
                    1,
                    ARD=False,
                ) + GPy.kern.Matern52(
                    1,
                    ARD=False,
                ) + GPy.kern.Matern32(
                    1,
                    ARD=False,
                )
        return kernel


def testimport() -> None:
    ''' For the testing import script. Simply prints hello world to the
    terminal.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    '''
    print('Hello world!')
    return
