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