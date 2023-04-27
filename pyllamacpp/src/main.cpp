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

#include "../llama.cpp/llama.h"
#include "main.h"



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static bool is_interacting = false;


py::function py_llama_progress_callback;

struct llama_context_wrapper {
    bool continue_gen = true;     // Continue text generation
    llama_context* ptr;
};

struct llama_context_wrapper llama_init_from_file_wrapper(const char * path_model, struct llama_context_params  params){
    struct llama_context * ctx = llama_init_from_file(path_model, params);
    struct llama_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}


void llama_free_wrapper(struct llama_context_wrapper * ctx_w){
    llama_free(ctx_w->ptr);
}

int llama_eval_wrapper(struct llama_context_wrapper * ctx_w,
               const llama_token * tokens,
               int   n_tokens,
               int   n_past,
               int   n_threads){
   struct llama_context * ctx = ctx_w->ptr;
   return llama_eval(ctx, tokens, n_tokens, n_past, n_threads);
}

std::vector<llama_token> llama_tokenize_wrapper(
        struct llama_context_wrapper * ctx_w,
        const std::string & text,
        bool   add_bos){

        struct llama_context * ctx = ctx_w->ptr;
        std::vector<llama_token> tokens((text.size() + (int)add_bos));
        int new_size = llama_tokenize(ctx, text.c_str(), tokens.data(), tokens.size(), add_bos);
        assert(new_size >= 0);
        tokens.resize(new_size);
        return tokens;
}

//py::array_t<llama_token> llama_tokenize_wrapper(
//        struct llama_context_wrapper * ctx_w,
//        const char * text,
//        int   n_max_tokens,
//        bool   add_bos){
//    struct llama_context * ctx = ctx_w->ptr;
//
//    py::array_t<llama_token> tokens;
//    py::buffer_info buf = tokens.request();
//    llama_token *tokens_ptr = static_cast<llama_token *>(buf.ptr);
//    llama_tokenize(ctx, text, tokens_ptr, n_max_tokens, add_bos);
//    return tokens;
//}



int llama_n_vocab_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_n_vocab(ctx);
}
int llama_n_ctx_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_n_ctx(ctx);
}
int llama_n_embd_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_n_embd(ctx);
}

float * llama_get_logits_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_get_logits(ctx);
}

float * llama_get_embeddings_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_get_embeddings(ctx);
}

const char * llama_token_to_str_wrapper(struct llama_context_wrapper * ctx_w, llama_token token){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_token_to_str(ctx, token);
}

llama_token llama_sample_top_p_top_k_wrapper(
        struct llama_context_wrapper * ctx_w,
        const llama_token * last_n_tokens_data,
        int   last_n_tokens_size,
        int   top_k,
        float   top_p,
        float   temp,
        float   repeat_penalty){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_sample_top_p_top_k(ctx, last_n_tokens_data, last_n_tokens_size, top_k, top_p, temp, repeat_penalty);
}

void llama_print_timings_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_print_timings(ctx);
}

void llama_reset_timings_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_reset_timings(ctx);
}

//void _llama_progress_callback(float progress, void *ctx){
//    struct llama_context_wrapper ctx_w;
//    ctx_w.ptr = ctx;
//    // call the python callback
////    py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
//    py_new_segment_callback(ctx_w, n_new, user_data);
//};


////////////////////////
std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
        default: return "To";
    }

    return "The";
}

// This is a function to return the tokens number
// Needed by front end to optimize data size
int llama_get_nb_tokens(struct llama_context_wrapper * ctx_w, std::string prompt){
    // tokenize the prompt
    auto embd_inp = ::llama_tokenize_wrapper(ctx_w, prompt, true);
    return embd_inp.size();
}

// quick and dirty implementation! just copied from main.cpp with some minor changes
// Needs lots of improvements
int llama_generate(struct llama_context_wrapper * ctx_w, gpt_params params, py::function new_text_callback, py::function grab_text_callback, bool verbose){

    // Set continue_gen to true
    ctx_w->continue_gen = true;

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                        "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    struct llama_context * ctx = ctx_w->ptr;

    // load the model
//    {
//        auto lparams = llama_context_default_params();
//
//        lparams.n_ctx      = params.n_ctx;
//        lparams.n_parts    = params.n_parts;
//        lparams.seed       = params.seed;
//        lparams.f16_kv     = params.memory_f16;
//        lparams.use_mlock  = params.use_mlock;
//
//        ctx = llama_init_from_file(params.model.c_str(), lparams);
//
//        if (ctx == NULL) {
//            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
//            return 1;
//        }
//    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<llama_token> tmp(params.n_batch, 0);
            llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }

        {
            const std::vector<llama_token> tmp = { 0, };
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);

        return 0;
    }

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize_wrapper(ctx_w, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    const auto inp_pfx = ::llama_tokenize_wrapper(ctx_w, "\n\n### Instruction:\n\n", true);
    const auto inp_sfx = ::llama_tokenize_wrapper(ctx_w, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive_start = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if reverse prompt or interactive start is specified
    if (params.antiprompt.size() != 0 || params.interactive_start) {
        params.interactive = true;
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize_wrapper(ctx_w, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        if (params.n_keep > 0) {
            fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

   if (params.interactive) {
//#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
//        struct sigaction sigint_action;
//        sigint_action.sa_handler = sigint_handler;
//        sigemptyset (&sigint_action.sa_mask);
//        sigint_action.sa_flags = 0;
//        sigaction(SIGINT, &sigint_action, NULL);
//#elif defined (_WIN32)
//        signal(SIGINT, sigint_handler);
//#endif
//
       fprintf(stderr, "%s: interactive mode on.\n", __func__);

       if (params.antiprompt.size()) {
           for (auto antiprompt : params.antiprompt) {
               fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
           }
       }

       if (!params.input_prefix.empty()) {
           fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
       }
   }
    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
            params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
                        #if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
                        " - Press Ctrl+C to interject at any time.\n"
                        #endif
                        " - Press Return to return control to LLaMa.\n"
                        " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = params.interactive_start;
    }

    bool is_antiprompt = false;
    bool input_noecho  = false;

    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;


    std::vector<llama_token> embd;

    while (n_remain != 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const int32_t top_k          = params.top_k;
            const float   top_p          = params.top_p;
            const float   temp           = params.temp;
            const float   repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (params.ignore_eos) {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx,
                                              last_n_tokens.data() + n_ctx - params.repeat_last_n,
                                              params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::llama_tokenize_wrapper(ctx_w, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (!input_noecho) {
            for (auto id : embd) {
//                printf("%s", llama_token_to_str(ctx, id));
                // If the host wants to stop generation, we should stop
                new_text_callback(llama_token_to_str(ctx, id));
                if(!ctx_w->continue_gen){
                    llama_print_timings(ctx);
                    return 0;
                }
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
//        if (!input_noecho && (int)embd_inp.size() == n_consumed) {
//            set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
//        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= n_consumed) {

            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string & antiprompt : params.antiprompt) {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                        is_interacting = true;
                        is_antiprompt = true;
//                        set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
                        fflush(stdout);
                        break;
                    }
                }
            }

            if (n_past > 0 && is_interacting) {
                // potentially set color to indicate we are taking user input
//                set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);

                if (params.instruct) {
                    printf("\n> ");
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                std::string line;
                bool another_line = true;
                do {
                    py::handle x = grab_text_callback();

                    if (x.is_none())
                    {
                        return 0;
                    }
                    else if (!py::isinstance<py::str>(x))
                    {
                        fprintf(stderr, "%s : input was not of type py::str. will ignore.\n", __func__);
                    }
                    else
                    {
                        line = x.cast<std::string>();
                        if (line.empty() || line.back() != '\\') {
                            another_line = false;
                        } else {
                            line.pop_back(); // Remove the continue character
                        }
                        buffer += line + '\n'; // Append the line to the result
                    }

                } while (another_line);

                // done taking input, reset color
//                set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    auto line_inp = ::llama_tokenize_wrapper(ctx_w, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    n_remain -= line_inp.size();
                }

                input_noecho = true; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    llama_print_timings(ctx);

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
        .def_readwrite("n_keep", &gpt_params::n_keep)
        .def_readwrite("model", &gpt_params::model)
        .def_readwrite("prompt", &gpt_params::prompt)
        .def_readwrite("use_color", &gpt_params::use_color)
        .def_readwrite("interactive", &gpt_params::interactive)
        .def_readwrite("interactive_start", &gpt_params::interactive_start)
        .def_readwrite("verbose_prompt", &gpt_params::verbose_prompt)
        .def_readwrite("antiprompt", &gpt_params::antiprompt);

    py::class_<llama_context_wrapper>(m,"llama_context")
        .def_readwrite("continue_gen", &llama_context_wrapper::continue_gen);

    py::class_<llama_token_data>(m,"llama_token_data")
        .def(py::init<>())
        .def_readwrite("id", &llama_token_data::id)
        .def_readwrite("p", &llama_token_data::p)
        .def_readwrite("plog", &llama_token_data::plog);
    
    py::class_<llama_context_params>(m,"llama_context_params")
        .def(py::init<>())
        .def_readwrite("n_ctx", &llama_context_params::n_ctx)
        .def_readwrite("n_parts", &llama_context_params::n_parts)
        .def_readwrite("seed", &llama_context_params::seed)
        .def_readwrite("f16_kv", &llama_context_params::f16_kv)
        .def_readwrite("logits_all", &llama_context_params::logits_all)
        .def_readwrite("vocab_only", &llama_context_params::vocab_only)
        .def_readwrite("use_mlock", &llama_context_params::use_mlock)
        .def_readwrite("embedding", &llama_context_params::embedding)
        .def_property("progress_callback", [](llama_context_params &self) {},
            [](llama_context_params &self, py::function callback) {
            py_llama_progress_callback = callback;
            self.progress_callback = [](float progress, void *ctx) {
//                struct llama_context_wrapper ctx_w;
//                ctx_w->ptr = ctx;
                py_llama_progress_callback(progress, ctx);
                };
        })
        .def_readwrite("progress_callback_user_data", &llama_context_params::progress_callback_user_data);
    m.def("llama_context_default_params", &llama_context_default_params);
    m.def("llama_init_from_file", &llama_init_from_file_wrapper);
    m.def("llama_free", &llama_free_wrapper);
    m.def("llama_model_quantize", &llama_model_quantize);
    m.def("llama_eval", &llama_eval_wrapper);
    m.def("llama_tokenize", &llama_tokenize_wrapper);
    m.def("llama_n_vocab", &llama_n_vocab_wrapper);
    m.def("llama_n_ctx", &llama_n_ctx_wrapper);
    m.def("llama_n_embd", &llama_n_embd_wrapper);
    m.def("llama_get_logits", &llama_get_logits_wrapper);
    m.def("llama_get_embeddings", &llama_get_embeddings_wrapper);
    m.def("llama_token_to_str", &llama_token_to_str_wrapper);

    m.def("llama_token_bos", &llama_token_bos);
    m.def("llama_token_eos", &llama_token_eos);

    m.def("llama_sample_top_p_top_k", &llama_sample_top_p_top_k_wrapper);

    m.def("llama_print_timings", &llama_print_timings_wrapper);
    m.def("llama_reset_timings", &llama_reset_timings_wrapper);

    m.def("llama_print_system_info", &llama_print_system_info);

    m.def("llama_get_nb_tokens", &llama_get_nb_tokens);
    m.def("llama_generate", &llama_generate);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
