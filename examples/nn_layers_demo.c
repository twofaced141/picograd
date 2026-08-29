#include <stdio.h>
#include <math.h>
#include "../src/autograd/autograd.h"
#include "../src/nn/nn.h"
#include "../src/core/tensor.h"

int main(void){
    printf("=== nn layers demo ===\n");

    /* Conv2d: 1x1 image, 1 in-channel, 1 out-channel, 1x1 kernel */
    printf("\n--- Conv2d ---");
    {
        pg_conv2d_layer *l = pg_conv2d_layer_new(1, 1, 1, 1, 1, 0);
        size_t shape[4] = {1, 1, 3, 3};
        float data[] = {1,2,3,4,5,6,7,8,9};
        pg_tensor *x = pg_tensor_from_data(4, shape, data);
        pg_tensor *y = pg_conv2d_layer_forward(l, x);
        printf("\nconv2d output shape [%zu,%zu,%zu,%zu]:\n", y->shape[0], y->shape[1], y->shape[2], y->shape[3]);
        pg_tensor_print(y, stdout);
        /* autograd path */
        pg_node *xn = pg_var_from_tensor(x, true);
        pg_node *on = pg_ag_conv2d_layer_forward(l, xn);
        pg_backward(on);
        printf("x grad shape [%zu]: ", xn->grad->numel);
        for(size_t i=0;i<xn->grad->numel;i++) printf("%g ", xn->grad->data[i]);
        printf("\n");
        pg_node_free(on); pg_node_free(xn);
        pg_tensor_free(y); pg_tensor_free(x);
        pg_conv2d_layer_free(l);
    }

    /* Embedding */
    printf("\n--- Embedding ---");
    {
        pg_embedding_layer *l = pg_embedding_layer_new(6, 3);
        size_t n = 4;
        float idata[] = {0, 3, 1, 4};
        pg_tensor *indices = pg_tensor_from_data(1, &n, idata);
        pg_tensor *y = pg_embedding_layer_forward(l, indices);
        printf("\nemb output shape [%zu,%zu]:\n", y->shape[0], y->shape[1]);
        pg_tensor_print(y, stdout);
        pg_node *yn = pg_ag_embedding_layer_forward(l, indices);
        pg_backward(yn);
        printf("emb weight grad shape [%zu]:\n", yn->grad->numel);
        pg_tensor_print(yn->grad, stdout);
        pg_node_free(yn); pg_tensor_free(y); pg_tensor_free(indices);
        pg_embedding_layer_free(l);
    }

    /* Dropout */
    printf("\n--- Dropout ---");
    {
        pg_dropout_layer *l = pg_dropout_layer_new(0.5f);
        float d[] = {1,2,3,4,5,6};
        pg_tensor *x = pg_tensor_from_data(1, (size_t[]){6}, d);
        pg_tensor *yt = pg_dropout_layer_forward(l, x, true);
        printf("\ndropout training output (must have zeros): ");
        for(size_t i=0;i<yt->numel;i++) printf("%g ", yt->data[i]);
        printf("\n");
        pg_tensor_free(yt);
        pg_node *xn = pg_var_from_tensor(x, true);
        pg_node *on = pg_ag_dropout_layer_forward(xn, 0.5f, true);
        pg_backward(on);
        printf("dropout training grad shape [%zu]: ", xn->grad->numel);
        for(size_t i=0;i<xn->grad->numel;i++) printf("%g ", xn->grad->data[i]);
        printf("\n");
        pg_node_free(on); pg_node_free(xn);

        /* eval: identity */
        pg_tensor *ye = pg_dropout_layer_forward(l, x, false);
        printf("dropout eval output (identity): ");
        for(size_t i=0;i<ye->numel;i++) printf("%g ", ye->data[i]);
        printf("\n");
        pg_tensor_free(ye);
        pg_node *xe = pg_var_from_tensor(x, true);
        pg_node *oe = pg_ag_dropout_layer_forward(xe, 0.5f, false);
        pg_backward(oe);
        printf("dropout eval grad (all 1s) shape [%zu]: ", xe->grad->numel);
        for(size_t i=0;i<xe->grad->numel;i++) printf("%g ", xe->grad->data[i]);
        printf("\n");
        pg_node_free(oe); pg_node_free(xe);
        pg_tensor_free(x);
        pg_dropout_layer_free(l);
    }

    printf("\nnn layers demo finished.\n");
    return 0;
}