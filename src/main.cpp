/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [abdeladim-s](https://github.com/abdeladim-s)
 * @date    2023
 * @brief   Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) using Pybind11
 *
 * @par
 * COPYRIGHT NOTICE: (c) 2023.
 ********************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "../llama.cpp/ggml.h"
#include "../llama.cpp/utils.h"
#include "main.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

#define QK 32

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


int llama_free(llama_model &model){
    ggml_free(model.ctx);
    return 0;
}


bool llama_eval_wrapper(
        const llama_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
        std::vector<float>         & embd_w
        ){
    // determine the required inference memory per token:
    std::vector<float> logits;
    size_t mem_per_token = 0;
    llama_eval(model, n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    return llama_eval(model, n_threads, n_past, embd_inp, embd_w, mem_per_token);
}

// quick and dirty implementation! just copied from main.cpp with some minor changes
// Needs lots of improvements
int llama_generate(gpt_params & params,llama_model & model,gpt_vocab & vocab, py::function new_text_callback, bool verbose){
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    // seed
    if (params.seed < 0) {
        params.seed = time(NULL);
        if(verbose){
            fprintf(stderr, "Seed = %d\n", params.seed);
        }
    }

    std::mt19937 rng(params.seed);

    // random seed if empty
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int n_past = 0;
    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, params.prompt, true);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    // tokenize the reverse prompt
    std::vector<gpt_vocab::id> antiprompt_inp = ::llama_tokenize(vocab, params.antiprompt, false);

    // some printing stuff
    {
        if(verbose){
            fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
            fprintf(stderr, "\n\n");
        }
    }

    std::vector<gpt_vocab::id> embd;
    std::vector<gpt_vocab::id> res_embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    llama_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int remaining_tokens = params.n_predict;
    int input_consumed = 0;
    bool input_noecho = false;

    // inference
    while (remaining_tokens > 0){
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!llama_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                fprintf(stderr, "Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();
        res_embd.clear();

        if (embd_inp.size() <= input_consumed) {
            // out of user input, sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
            res_embd.push_back(id);

            // decrement remaining sampling budget
            --remaining_tokens;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while (embd_inp.size() > input_consumed) {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if (embd.size() > params.n_batch) {
                    break;
                }
            }

        }

        for (auto id : res_embd) {
//                printf("%s", vocab.id_to_token[id].c_str());
            if(new_text_callback != py::none()){
                try {
                    new_text_callback(vocab.id_to_token[id].c_str());
                }
                catch (...)
                {
                    fprintf(stderr, "Error parsing token: \n", id);
                }

            }
        }

        if (params.interactive) {
            // check for reverse prompt
            if (antiprompt_inp.size() &&
                std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(), last_n_tokens.rbegin())) {
                // reverse prompt found
                return 0;
            }
        }

        // end of text token
        if (embd.back() == 2) {
//            fprintf(stderr, " [end of text]\n");
            break;
        }

    }




    // report timing
    if(verbose){
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        fprintf(stderr, "%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        fprintf(stderr, "%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }



    return 0;
}

bool llama_model_quantize(const std::string & fname_inp, const std::string & fname_out, int itype) {
    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
        case 2: type = GGML_TYPE_Q4_0; break;
        case 3: type = GGML_TYPE_Q4_1; break;
        default: fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype); return 1;
    };

    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1) {
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, type);
        return false;
    }

    gpt_vocab vocab;

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *) &magic, sizeof(magic));
    }

    llama_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //finp.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        finp.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        finp.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        finp.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        finp.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        finp.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        finp.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_mult  = %d\n", __func__, hparams.n_mult);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);

        fout.write((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fout.write((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fout.write((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fout.write((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fout.write((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fout.write((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fout.write((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fout.write((char *) &itype,           sizeof(hparams.f16));
    }

    // load vocab
    {
        const int32_t n_vocab = hparams.n_vocab;

        if (n_vocab != hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word.resize(len);
            finp.read ((char *) word.data(), len);
            fout.write((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t>     data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float>       data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (finp.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read (&name[0], length);

            {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                    ".*weight",
            };

            bool quantize = false;
            for (const auto & s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = true;
                    break;
                }
            }

            // quantize only 2D tensors
            quantize &= (n_dims == 2);

            if (quantize) {
                if (ftype != 0 && ftype != 1) {
                    fprintf(stderr, "%s: unsupported ftype %d for integer quantization\n", __func__, ftype);
                    return false;
                }

                if (ftype == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = itype;
            } else {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements*bpe);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            if (quantize) {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (type) {
                    case GGML_TYPE_Q4_0:
                    {
                        cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], QK, hist_cur.data());
                    } break;
                    case GGML_TYPE_Q4_1:
                    {
                        cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], QK, hist_cur.data());
                    } break;
                    default:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                        return false;
                    }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
                for (int i = 0; i < hist_cur.size(); ++i) {
                    hist_all[i] += hist_cur[i];
                }

                for (int i = 0; i < hist_cur.size(); ++i) {
                    printf("%5.3f ", hist_cur[i] / (float)nelements);
                }
                printf("\n");
            } else {
                printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

        {
            int64_t sum_all = 0;
            for (int i = 0; i < hist_all.size(); ++i) {
                sum_all += hist_all[i];
            }

            printf("%s: hist: ", __func__);
            for (int i = 0; i < hist_all.size(); ++i) {
                printf("%5.3f ", hist_all[i] / (float)sum_all);
            }
            printf("\n");
        }
    }

    finp.close();
    fout.close();

    return true;
}


// quantize a model
int llama_quantize(const std::string & fname_inp, const std::string & fname_out, int itype) {

    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

//    const std::string fname_inp = fname_inp;
//    const std::string fname_out = fname_out;
//
//    const int itype = itype;

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!llama_model_quantize(fname_inp, fname_out, itype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}

PYBIND11_MODULE(_pyllamacpp, m) {
    m.doc() = R"pbdoc(
        PyLlamaCpp: Python binding to llama.cpp
        -----------------------

        .. currentmodule:: _pyllamacpp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    ///// utils.h

    py::class_<gpt_params>(m,"gpt_params" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        .def_readwrite("seed", &gpt_params::seed)
        .def_readwrite("n_threads", &gpt_params::n_threads)
        .def_readwrite("n_predict", &gpt_params::n_predict)
        .def_readwrite("repeat_last_n", &gpt_params::repeat_last_n)
        .def_readwrite("top_k", &gpt_params::top_k)
        .def_readwrite("top_p", &gpt_params::top_p)
        .def_readwrite("temp", &gpt_params::temp)
        .def_readwrite("repeat_penalty", &gpt_params::repeat_penalty)
        .def_readwrite("n_batch", &gpt_params::n_batch)
        .def_readwrite("model", &gpt_params::model)
        .def_readwrite("prompt", &gpt_params::prompt)
        .def_readwrite("use_color", &gpt_params::use_color)
        .def_readwrite("interactive", &gpt_params::interactive)
        .def_readwrite("interactive_start", &gpt_params::interactive_start)
        .def_readwrite("antiprompt", &gpt_params::antiprompt)
        ;

    py::class_<gpt_vocab>(m, "gpt_vocab")
            .def(py::init<>())
            .def_readwrite("token_to_id", &gpt_vocab::token_to_id)
            .def_readwrite("id_to_token", &gpt_vocab::id_to_token)
            ;

    m.def("json_parse", &json_parse);
    m.def("gpt_tokenize", &gpt_tokenize);
    m.def("llama_tokenize", &llama_tokenize);
    m.def("gpt_vocab_init", &gpt_vocab_init);
    m.def("llama_sample_top_p_top_k", &llama_sample_top_p_top_k);
    m.def("sample_top_k", &sample_top_k);
    m.def("ggml_quantize_q4_0", &ggml_quantize_q4_0);
    m.def("ggml_quantize_q4_1", &ggml_quantize_q4_1);
    m.def("gpt_random_prompt", &gpt_random_prompt);

    // main.cpp

    py::class_<llama_hparams>(m, "llama_hparams")
            .def(py::init<>())
            .def_readwrite("n_vocab", &llama_hparams::n_vocab)
            .def_readwrite("n_ctx", &llama_hparams::n_ctx)
            .def_readwrite("n_embd", &llama_hparams::n_embd)
            .def_readwrite("n_mult", &llama_hparams::n_mult)
            .def_readwrite("n_head", &llama_hparams::n_head)
            .def_readwrite("n_layer", &llama_hparams::n_layer)
            .def_readwrite("n_rot", &llama_hparams::n_rot)
            .def_readwrite("f16", &llama_hparams::f16)
            ;

     py::class_<llama_model>(m, "llama_model")
            .def(py::init<>())
            ;


    m.def("llama_print_system_info", &llama_print_system_info);
    m.def("llama_model_load", &llama_model_load);
    m.def("llama_eval", &llama_eval);
    m.def("llama_eval_wrapper", &llama_eval_wrapper);

    m.def("llama_generate", &llama_generate);
    m.def("llama_free", &llama_free);
    m.def("llama_quantize", &llama_quantize);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
