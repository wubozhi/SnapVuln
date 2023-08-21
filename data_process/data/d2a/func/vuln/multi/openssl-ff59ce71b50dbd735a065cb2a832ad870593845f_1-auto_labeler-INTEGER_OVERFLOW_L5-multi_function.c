static int pkcs12_gen_mac(PKCS12 *p12, const char *pass, int passlen,
                          unsigned char *mac, unsigned int *maclen,
                          int (*pkcs12_key_gen)(const char *pass, int passlen,
                                                unsigned char *salt, int slen,
                                                int id, int iter, int n,
                                                unsigned char *out,
                                                const EVP_MD *md_type))
{
    const EVP_MD *md_type;
    HMAC_CTX *hmac = NULL;
    unsigned char key[EVP_MAX_MD_SIZE], *salt;
    int saltlen, iter;
    int md_size = 0;
    int md_type_nid;
    const X509_ALGOR *macalg;
    const ASN1_OBJECT *macoid;

    if (pkcs12_key_gen == NULL)
        pkcs12_key_gen = PKCS12_key_gen_utf8;

    if (!PKCS7_type_is_data(p12->authsafes)) {
        PKCS12err(PKCS12_F_PKCS12_GEN_MAC, PKCS12_R_CONTENT_TYPE_NOT_DATA);
        return 0;
    }

    salt = p12->mac->salt->data;
    saltlen = p12->mac->salt->length;
    if (!p12->mac->iter)
        iter = 1;
    else
        iter = ASN1_INTEGER_get(p12->mac->iter);
    X509_SIG_get0(p12->mac->dinfo, &macalg, NULL);
    X509_ALGOR_get0(&macoid, NULL, NULL, macalg);
    if ((md_type = EVP_get_digestbyobj(macoid)) == NULL) {
        PKCS12err(PKCS12_F_PKCS12_GEN_MAC, PKCS12_R_UNKNOWN_DIGEST_ALGORITHM);
        return 0;
    }
    md_size = EVP_MD_size(md_type);
    md_type_nid = EVP_MD_type(md_type);
    if (md_size < 0)
        return 0;
    if ((md_type_nid == NID_id_GostR3411_94
         || md_type_nid == NID_id_GostR3411_2012_256
         || md_type_nid == NID_id_GostR3411_2012_512)
        && !getenv("LEGACY_GOST_PKCS12")) {
        md_size = TK26_MAC_KEY_LEN;
        if (!pkcs12_gen_gost_mac_key(pass, passlen, salt, saltlen, iter,
                                     md_size, key, md_type)) {
            PKCS12err(PKCS12_F_PKCS12_GEN_MAC, PKCS12_R_KEY_GEN_ERROR);
            return 0;
        }
    } else
        if (!(*pkcs12_key_gen)(pass, passlen, salt, saltlen, PKCS12_MAC_ID,
                               iter, md_size, key, md_type)) {
        PKCS12err(PKCS12_F_PKCS12_GEN_MAC, PKCS12_R_KEY_GEN_ERROR);
        return 0;
    }
    if ((hmac = HMAC_CTX_new()) == NULL
        || !HMAC_Init_ex(hmac, key, md_size, md_type, NULL)
        || !HMAC_Update(hmac, p12->authsafes->d.data->data,
                        p12->authsafes->d.data->length)
        || !HMAC_Final(hmac, mac, maclen)) {
        HMAC_CTX_free(hmac);
        return 0;
    }
    HMAC_CTX_free(hmac);
    return 1;
}

HMAC_CTX *HMAC_CTX_new(void)
{
    HMAC_CTX *ctx = OPENSSL_zalloc(sizeof(HMAC_CTX));

    if (ctx != NULL) {
        if (!HMAC_CTX_reset(ctx)) {
            HMAC_CTX_free(ctx);
            return NULL;
        }
    }
    return ctx;
}

int HMAC_CTX_reset(HMAC_CTX *ctx)
{
    hmac_ctx_cleanup(ctx);
    if (!hmac_ctx_alloc_mds(ctx)) {
        hmac_ctx_cleanup(ctx);
        return 0;
    }
    return 1;
}

static void hmac_ctx_cleanup(HMAC_CTX *ctx)
{
    EVP_MD_CTX_reset(ctx->i_ctx);
    EVP_MD_CTX_reset(ctx->o_ctx);
    EVP_MD_CTX_reset(ctx->md_ctx);
    ctx->md = NULL;
    ctx->key_length = 0;
    OPENSSL_cleanse(ctx->key, sizeof(ctx->key));
}

int HMAC_Init_ex(HMAC_CTX *ctx, const void *key, int len,
                 const EVP_MD *md, ENGINE *impl)
{
    int rv = 0;
    int i, j, reset = 0;
    unsigned char pad[HMAC_MAX_MD_CBLOCK];

    /* If we are changing MD then we must have a key */
    if (md != NULL && md != ctx->md && (key == NULL || len < 0))
        return 0;

    if (md != NULL) {
        reset = 1;
        ctx->md = md;
    } else if (ctx->md) {
        md = ctx->md;
    } else {
        return 0;
    }

    if (key != NULL) {
        reset = 1;
        j = EVP_MD_block_size(md);
        if (!ossl_assert(j <= (int)sizeof(ctx->key)))
            return 0;
        if (j < len) {
            if (!EVP_DigestInit_ex(ctx->md_ctx, md, impl)
                    || !EVP_DigestUpdate(ctx->md_ctx, key, len)
                    || !EVP_DigestFinal_ex(ctx->md_ctx, ctx->key,
                                           &ctx->key_length))
                return 0;
        } else {
            if (len < 0 || len > (int)sizeof(ctx->key))
                return 0;
            memcpy(ctx->key, key, len);
            ctx->key_length = len;
        }
        if (ctx->key_length != HMAC_MAX_MD_CBLOCK)
            memset(&ctx->key[ctx->key_length], 0,
                   HMAC_MAX_MD_CBLOCK - ctx->key_length);
    }

    if (reset) {
        for (i = 0; i < HMAC_MAX_MD_CBLOCK; i++)
            pad[i] = 0x36 ^ ctx->key[i];
        if (!EVP_DigestInit_ex(ctx->i_ctx, md, impl)
                || !EVP_DigestUpdate(ctx->i_ctx, pad, EVP_MD_block_size(md)))
            goto err;

        for (i = 0; i < HMAC_MAX_MD_CBLOCK; i++)
            pad[i] = 0x5c ^ ctx->key[i];
        if (!EVP_DigestInit_ex(ctx->o_ctx, md, impl)
                || !EVP_DigestUpdate(ctx->o_ctx, pad, EVP_MD_block_size(md)))
            goto err;
    }
    if (!EVP_MD_CTX_copy_ex(ctx->md_ctx, ctx->i_ctx))
        goto err;
    rv = 1;
 err:
    if (reset)
        OPENSSL_cleanse(pad, sizeof(pad));
    return rv;
}

