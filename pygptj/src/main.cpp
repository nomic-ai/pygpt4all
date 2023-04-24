/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [abdeladim-s](https://github.com/abdeladim-s)
 * @date    2023
 * @brief   Python bindings for GPT-J Language model based on [ggml](https://github.com/ggerganov/ggml)
 * @par     ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov,
            please see [ggml License](./GGML_LICENSE)
 ********************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "utils.cpp"
#include "gptj.cpp"
#include "main.h"



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


PYBIND11_MODULE(_pygptj, m) {
    m.doc() = R"pbdoc(
        PyGPT-J: Python binding to GPT-J
        -----------------------

        .. currentmodule:: _pygptj

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py::class_<gpt_params>(m,"gptj_gpt_params" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        .def_readwrite("seed", &gpt_params::seed)
        .def_readwrite("n_threads", &gpt_params::n_threads)
        .def_readwrite("n_predict", &gpt_params::n_predict)
        .def_readwrite("top_k", &gpt_params::top_k)
        .def_readwrite("top_p", &gpt_params::top_p)
        .def_readwrite("temp", &gpt_params::temp)
        .def_readwrite("n_batch", &gpt_params::n_batch)
        .def_readwrite("model", &gpt_params::model)
        .def_readwrite("prompt", &gpt_params::prompt)
        ;

    py::class_<gptj_hparams>(m,"gptj_hparams" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        ;
    py::class_<gptj_model>(m,"gptj_model" /*,py::dynamic_attr()*/)
        .def(py::init<>())
    ;
 py::class_<gpt_vocab>(m,"gpt_vocab" /*,py::dynamic_attr()*/)
        .def(py::init<>())
    ;

    m.def("gptj_model_load", &gptj_model_load);
    m.def("gptj_eval", &gptj_eval);
    m.def("gptj_free", &gptj_free);
    m.def("gpt_sample_top_k_top_p", &gpt_sample_top_k_top_p);
    m.def("gpt_tokenize", &gpt_tokenize);

    m.def("gptj_generate", &gptj_generate);




#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
