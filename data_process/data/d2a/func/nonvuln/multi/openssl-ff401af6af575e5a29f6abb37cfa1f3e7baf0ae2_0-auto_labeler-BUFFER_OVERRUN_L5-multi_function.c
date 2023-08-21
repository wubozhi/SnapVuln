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

void *CRYPTO_malloc(int num, const char *file, int line)
{
    void *ret = NULL;

    if (num <= 0)
        return NULL;

    if (allow_customize)
        allow_customize = 0;
    if (malloc_debug_func != NULL) {
        if (allow_customize_debug)
            allow_customize_debug = 0;
        malloc_debug_func(NULL, num, file, line, 0);
    }
    ret = malloc_ex_func(num, file, line);
#ifdef LEVITTE_DEBUG_MEM
    fprintf(stderr, "LEVITTE_DEBUG_MEM:         > 0x%p (%d)\n", ret, num);
#endif
    if (malloc_debug_func != NULL)
        malloc_debug_func(ret, num, file, line, 1);

#ifndef OPENSSL_CPUID_OBJ
    /*
     * Create a dependency on the value of 'cleanse_ctr' so our memory
     * sanitisation function can't be optimised out. NB: We only do this for
     * >2Kb so the overhead doesn't bother us.
     */
    if (ret && (num > 2048)) {
        extern unsigned char cleanse_ctr;
        ((unsigned char *)ret)[0] = cleanse_ctr;
    }
#endif

    return ret;
}

static int getModulusAndExponent(const unsigned char *token,
                                 long *exponentLength,
                                 unsigned char *exponent, long *modulusLength,
                                 long *modulusFieldLength,
                                 unsigned char *modulus)
{
    unsigned long len;

    if (*token++ != (char)0x1E) /* internal PKA token? */
        return 0;

    if (*token++)               /* token version must be zero */
        return 0;

    len = *token++;
    len = len << 8;
    len |= (unsigned char)*token++;

    token += 4;                 /* skip reserved bytes */

    if (*token++ == (char)0x04) {
        if (*token++)           /* token version must be zero */
            return 0;

        len = *token++;
        len = len << 8;
        len |= (unsigned char)*token++;

        token += 2;             /* skip reserved section */

        len = *token++;
        len = len << 8;
        len |= (unsigned char)*token++;

        *exponentLength = len;

        len = *token++;
        len = len << 8;
        len |= (unsigned char)*token++;

        *modulusLength = len;

        len = *token++;
        len = len << 8;
        len |= (unsigned char)*token++;

        *modulusFieldLength = len;

        memcpy(exponent, token, *exponentLength);
        token += *exponentLength;

        memcpy(modulus, token, *modulusFieldLength);
        return 1;
    }
    return 0;
}

