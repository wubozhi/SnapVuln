static EVP_PKEY *ibm_4758_load_pubkey(ENGINE *e, const char *key_id,
                                      UI_METHOD *ui_method,
                                      void *callback_data)
{
    RSA *rtmp = NULL;
    EVP_PKEY *res = NULL;
    unsigned char *keyToken = NULL;
    long keyTokenLength = MAX_CCA_PKA_TOKEN_SIZE;
    long returnCode;
    long reasonCode;
    long exitDataLength = 0;
    long ruleArrayLength = 0;
    unsigned char exitData[8];
    unsigned char ruleArray[8];
    unsigned char keyLabel[64];
    unsigned long keyLabelLength = strlen(key_id);
    unsigned char modulus[512];
    long modulusFieldLength = sizeof(modulus);
    long modulusLength = 0;
    unsigned char exponent[512];
    long exponentLength = sizeof(exponent);

    if (keyLabelLength > sizeof(keyLabel)) {
        CCA4758err(CCA4758_F_IBM_4758_LOAD_PUBKEY,
                   CCA4758_R_SIZE_TOO_LARGE_OR_TOO_SMALL);
        return NULL;
    }

    memset(keyLabel, ' ', sizeof(keyLabel));
    memcpy(keyLabel, key_id, keyLabelLength);

    keyToken = OPENSSL_malloc(MAX_CCA_PKA_TOKEN_SIZE + sizeof(long));
    if (!keyToken) {
        CCA4758err(CCA4758_F_IBM_4758_LOAD_PUBKEY, ERR_R_MALLOC_FAILURE);
        goto err;
    }

    keyRecordRead(&returnCode, &reasonCode, &exitDataLength, exitData,
                  &ruleArrayLength, ruleArray, keyLabel, &keyTokenLength,
                  keyToken + sizeof(long));

    if (returnCode) {
        CCA4758err(CCA4758_F_IBM_4758_LOAD_PUBKEY, ERR_R_MALLOC_FAILURE);
        goto err;
    }

    if (!getModulusAndExponent(keyToken + sizeof(long), &exponentLength,
                               exponent, &modulusLength, &modulusFieldLength,
                               modulus)) {
        CCA4758err(CCA4758_F_IBM_4758_LOAD_PUBKEY,
                   CCA4758_R_FAILED_LOADING_PUBLIC_KEY);
        goto err;
    }

    (*(long *)keyToken) = keyTokenLength;
    rtmp = RSA_new_method(e);
    RSA_set_ex_data(rtmp, hndidx, (char *)keyToken);
    rtmp->e = BN_bin2bn(exponent, exponentLength, NULL);
    rtmp->n = BN_bin2bn(modulus, modulusFieldLength, NULL);
    rtmp->flags |= RSA_FLAG_EXT_PKEY;
    res = EVP_PKEY_new();
    EVP_PKEY_assign_RSA(res, rtmp);

    return res;
 err:
    if (keyToken)
        OPENSSL_free(keyToken);
    return NULL;
}