#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "malloc.h"
#include "fips202.h"
#include "aesctr.h"
#include "data.h"
#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "sparsemat.h"
#include "jlproj.h"
#include "labrador.h"
#include "chihuahua.h"

polx *init_sparsecnst_half(sparsecnst *cnst, size_t r, size_t nz, size_t buflen, size_t deg,
                           int quadratic, int homogeneous)
{
  void *buf;

  cnst->deg = deg;
  cnst->nz = nz;
  cnst->a->len = 0;
  if(quadratic) {
    buf = _malloc((r*r+r)*sizeof(size_t));
    cnst->a->rows = (size_t*)buf;
    cnst->a->cols = (size_t*)buf + (r*r+r)/2;

    buf = _aligned_alloc(64,r*sizeof(polx));
    cnst->a->coeffs = (polx*)buf;
  }
  else {
    cnst->a->rows = NULL;
    cnst->a->cols = NULL;
    cnst->a->coeffs = NULL;
  }

  if(!homogeneous)
    buflen += MAX(1,deg);

  buf = _malloc(4*nz*sizeof(size_t) + nz*sizeof(polx*) + 63 + buflen*sizeof(polx));
  cnst->idx = (size_t*)buf;
  cnst->off = &cnst->idx[nz];
  cnst->len = &cnst->off[nz];
  cnst->mult = &cnst->len[nz];
  cnst->phi = (polx**)&cnst->mult[nz];
  buf = (void*)(((uintptr_t)&cnst->phi[nz] + 63) & -64ULL);

  if(homogeneous)
    cnst->b = NULL;
  else {
    cnst->b = buf;
    buf = (polx*)buf + MAX(1,deg);
  }

  return buf;
}

void init_sparsecnst_raw(sparsecnst *cnst, size_t r, size_t nz, const size_t idx[nz], const size_t n[nz], size_t deg,
                         int quadratic, int homogeneous)
{
  size_t i;
  size_t buflen;
  size_t elen[nz];
  polx *buf;

  buflen = 0;
  for(i=0;i<nz;i++) {
    elen[i] = extlen(n[i]*deg,deg);
    buflen += elen[i];
  }

  buf = init_sparsecnst_half(cnst,r,nz,buflen,deg,quadratic,homogeneous);
  for(i=0;i<nz;i++) {
    cnst->idx[i] = idx[i];
    cnst->off[i] = 0;
    cnst->len[i] = n[i]*deg;
    cnst->mult[i] = 1;
    cnst->phi[i] = (polx*)buf;
    buf += elen[i];
  }
}

int set_sparsecnst_raw(sparsecnst *cnst, uint8_t h[16], size_t nz, const size_t idx[nz],
                       const size_t n[nz], size_t deg, int64_t *phi, int64_t *b)
{
  size_t j,k;
  polz t[deg];
  __attribute__((aligned(16)))
  uint8_t hashbuf[deg*N*QBYTES];
  shake128incctx shakectx;

  if(nz != cnst->nz) {
    fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in number of affected vectors\n");
    return 1;
  }
  if(deg != cnst->deg) {
    fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in extension degree\n");
    return 2;
  }
  if((b && !cnst->b) || (!b && cnst->b)) {
    fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in homogeneity\n");
    return 3;
  }
  for(j=0;j<nz;j++) {
    if(idx[j] != cnst->idx[j]) {
      fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in index of %zu-th affected vector\n",j);
      return 4;
    }
    if(cnst->off[j])  //FIXME
      return 5;
    if(cnst->mult[j] != 1)
      return 6;
  }

  cnst->a->len = 0;

  shake128_inc_init(&shakectx);
  shake128_inc_absorb(&shakectx,h,16);

  if(b) {
    polzvec_fromint64vec(t,1,deg,b);
    polzvec_topolxvec(cnst->b,t,deg);
    polzvec_bitpack(hashbuf,t,deg);
    shake128_inc_absorb(&shakectx,hashbuf,deg*N*QBYTES);
  }

  for(j=0;j<nz;j++) {
    for(k=0;k<n[j];k++) {
      polzvec_fromint64vec(t,1,deg,phi);
      polzvec_topolxvec(&cnst->phi[j][k*deg],t,deg);
      polzvec_bitpack(hashbuf,t,deg);
      shake128_inc_absorb(&shakectx,hashbuf,deg*N*QBYTES);
      phi += deg*N;
    }
  }

  shake128_inc_finalize(&shakectx);
  shake128_inc_squeeze(h,16,&shakectx);
  return 0;
}

void free_sparsecnst(sparsecnst *cnst) {
  free(cnst->idx);
  free(cnst->a->rows);
  free(cnst->a->coeffs);
  cnst->idx = cnst->a->rows = NULL;
  cnst->a->coeffs = NULL;
}

void sparsecnst_eval(polx *b, const sparsecnst *cnst, polx *sx[], const witness *wt) {
  const size_t r = wt->r;
  const size_t *n = wt->n;

  size_t i,j,k;
  size_t deg2 = MAX(1,cnst->deg);
  polx t[deg2];

  if(!*sx) {
    k = 0;
    for(i=0;i<r;i++)
      k += n[i];
    sx[0] = _aligned_alloc(64,k*sizeof(polx));
    for(i=1;i<r;i++)
      sx[i] = &sx[i-1][n[i-1]];
    for(i=0;i<r;i++)
      polxvec_frompolyvec(sx[i],wt->s[i],n[i]);
  }

  polxvec_setzero(b,deg2);

  /* quadratic term */
  for(i=0;i<cnst->a->len;i++) {
    j = cnst->a->rows[i];
    k = cnst->a->cols[i];
    polxvec_sprod(t,sx[j],sx[k],MIN(n[j],n[k]));  // TODO: store?
    polx_mul(t,t,&cnst->a->coeffs[i]);
    polx_add(b,b,t);
  }

  /* linear term */
  for(i=0;i<cnst->nz;i++) {
    j = cnst->idx[i];
    k = cnst->off[i];
    polxvec_mul_extension(t,cnst->phi[i],&sx[j][k],cnst->len[i],deg2/cnst->mult[i],cnst->mult[i]);
    polxvec_add(b,b,t,deg2);
  }
}

int sparsecnst_check(const sparsecnst *cnst, polx *sx[], const witness *wt) {
  int ret;
  size_t deg2 = MAX(1,cnst->deg);
  polx b[deg2];

  sparsecnst_eval(b,cnst,sx,wt);
  if(cnst->b)
    polxvec_sub(b,b,cnst->b,deg2);

  if(cnst->deg)
    ret = polxvec_iszero(b,deg2);
  else
    ret = polx_iszero_constcoeff(b);

  return ret;
}

/* One (phi buffer, witness slice) block of one constraint, ready to be evaluated
 * in an order of our choosing. */
typedef struct {
  const polx *phi;
  const polx *s;
  polx *acc;
  size_t len;
  size_t deg2;
  size_t mult;
} blockent;

static int blockent_cmp(const void *x, const void *y) {
  const polx *const a = ((const blockent*)x)->phi;
  const polx *const b = ((const blockent*)y)->phi;
  return (a > b) - (a < b);
}

/* Checks every constraint at once. Constraints alias each other's phi buffers, so
 * the blocks are evaluated phi-major: each distinct buffer is streamed once
 * instead of once per referencing constraint. Returns k if all hold, else the
 * index of the first that does not. */
size_t sparsecnst_check_batch(const sparsecnst *cnst, size_t k, polx *sx[], const witness *wt) {
  const size_t r = wt->r;
  const size_t *n = wt->n;

  size_t i,j,u,v,ne,tot,ret;
  polx *acc;
  blockent *ent;

  if(!*sx) {
    u = 0;
    for(i=0;i<r;i++)
      u += n[i];
    sx[0] = _aligned_alloc(64,u*sizeof(polx));
    for(i=1;i<r;i++)
      sx[i] = &sx[i-1][n[i-1]];
    for(i=0;i<r;i++)
      polxvec_frompolyvec(sx[i],wt->s[i],n[i]);
  }

  tot = 0;
  ne = 0;
  for(i=0;i<k;i++) {
    tot += MAX(1,cnst[i].deg);
    ne += cnst[i].nz;
  }
  acc = _aligned_alloc(64,MAX(tot,1)*sizeof(polx));
  polxvec_setzero(acc,tot);
  ent = _malloc(MAX(ne,1)*sizeof(blockent));

  ne = 0;
  tot = 0;
  for(i=0;i<k;i++) {
    const size_t deg2 = MAX(1,cnst[i].deg);
    for(j=0;j<cnst[i].nz;j++) {
      ent[ne].phi = cnst[i].phi[j];
      ent[ne].s = &sx[cnst[i].idx[j]][cnst[i].off[j]];
      ent[ne].acc = &acc[tot];
      ent[ne].len = cnst[i].len[j];
      ent[ne].deg2 = deg2;
      ent[ne].mult = cnst[i].mult[j];
      ne += 1;
    }
    tot += deg2;
  }

  qsort(ent,ne,sizeof(blockent),blockent_cmp);
  for(i=0;i<ne;i++) {
    polx t[ent[i].deg2];
    polxvec_mul_extension(t,ent[i].phi,ent[i].s,ent[i].len,ent[i].deg2/ent[i].mult,ent[i].mult);
    polxvec_add(ent[i].acc,ent[i].acc,t,ent[i].deg2);
  }

  ret = k;
  tot = 0;
  for(i=0;i<k;i++) {
    const size_t deg2 = MAX(1,cnst[i].deg);
    polx *b = &acc[tot];
    polx t[deg2];
    tot += deg2;

    for(j=0;j<cnst[i].a->len;j++) {
      u = cnst[i].a->rows[j];
      v = cnst[i].a->cols[j];
      polxvec_sprod(t,sx[u],sx[v],MIN(n[u],n[v]));
      polx_mul(t,t,&cnst[i].a->coeffs[j]);
      polx_add(b,b,t);
    }
    if(cnst[i].b)
      polxvec_sub(b,b,cnst[i].b,deg2);

    if(!(cnst[i].deg ? polxvec_iszero(b,deg2) : polx_iszero_constcoeff(b))) {
      ret = i;
      break;
    }
  }

  free(ent);
  free(acc);
  return ret;
}

int init_prncplstmnt_raw(prncplstmnt *st, size_t r, const size_t n[r],
                         uint64_t betasq, size_t k, int quadratic)
{
  size_t i;
  void *buf;

  if(betasq > JLMAXNORMSQ) {
    fprintf(stderr,"ERROR in init_prcplstmnt_raw(): Total witness norm too big for JL projection\n");
    return 1;
  }

  buf = _malloc(r*sizeof(size_t) + k*sizeof(sparsecnst));
  st->n = (size_t*)buf;
  st->cnst = (sparsecnst*)&st->n[r];

  st->r = r;
  st->k = k;
  st->quadratic = quadratic;
  st->betasq = betasq;
  for(i=0;i<r;i++)
    st->n[i] = n[i];
  for(i=0;i<k;i++)
    st->cnst[i].idx = NULL;

  memset(st->h,0,16);
  return 0;
}

int set_prncplstmnt_lincnst_raw(prncplstmnt *st, size_t i, size_t nz, const size_t idx[nz],
                                const size_t n[nz], size_t deg, int64_t *phi, int64_t *b)
{
  size_t j,k;

  if(i >= st->k) {
    fprintf(stderr,"ERROR in set_prcplstmnt_lincnst_raw(): Constraint %zu does not exist\n",i);
    return 1;
  }

  sparsecnst *cnst = &st->cnst[i];
  if(cnst->idx) {
    fprintf(stderr,"ERROR in set_prncplstmnt_lincnst_raw(): Constraint has already been set\n");
    return 2;
  }
  for(j=0;j<nz;j++) {
    k = idx[j];
    if(k >= st->r) {
      fprintf(stderr,"ERROR in set_prncplstmnt_lincnst_raw(): Witness vector %zu does not exist\n",k);
      return 3;
    }
    if(n[j]*deg != st->n[k]) {
      fprintf(stderr,"ERROR in set_prncplstmnt_lincnst_raw(): Mismatch in witness vector length\n");
      return 4;
    }
  }

  init_sparsecnst_raw(cnst,st->r,nz,idx,n,deg,0,b == NULL);
  (void)set_sparsecnst_raw(cnst,st->h,nz,idx,n,deg,phi,b);  // errors can not happen
  return 0;
}

void free_prncplstmnt(prncplstmnt *st) {
  size_t i;

  if(!st->n) return;
  for(i=0;i<st->k;i++)
    free_sparsecnst(&st->cnst[i]);
  free(st->n);  // one buffer for n,cnst
  st->n = NULL;
}

void print_prncplstmnt_pp(const prncplstmnt *st) {
  size_t i;

  printf("Chihuahua statement ");
  for(i=0;i<5;i++)
    printf("%02hhX",st->h[i]);
  printf(":\n");

  printf("  Witness multiplicity: %zu\n",st->r);
  printf("  Witness ranks: ");
  for(i=0;i<st->r;i++) {
    printf("%zu",st->n[i]);
    if(i<st->r-1) printf(", ");
  }
  printf("\n");

  printf("  Number of dot-product constraints: %zu\n",st->k);
  printf("  Norm constraint: %.2f\n",sqrt(st->betasq));
  printf("\n");
}

#define AGGCHUNK 16

/* A destination run written by a constant-term constraint. */
typedef struct {
  polx *p;
  size_t len;
} phirun;

static int phirun_cmp(const void *x, const void *y) {
  polx *const a = ((const phirun*)x)->p;
  polx *const b = ((const phirun*)y)->p;
  return (a > b) - (a < b);
}

void collaps_sparsecnst(constraint *ocnst, statement *ost, const proof *pi, const sparsecnst *icnst, size_t k) {
  size_t i,j,m,u,v,nc,nb;
  const size_t n = ost->n;
  int64_t s;
  polx (*phi)[n];
  polx **dbase;
  phirun *cov;

  m = 0;
  nb = 0;
  for(i=0;i<k;i++)
    if(icnst[i].deg == 0) {
      m += 1;
      nb += icnst[i].nz;
    }

  __attribute__((aligned(16)))
  uint8_t hashbuf[16+m*QBYTES];

  shake128(hashbuf,sizeof(hashbuf),ost->h,16);
  memcpy(ost->h,hashbuf,16);

  /* Destination base of every witness index, walked once instead of once per block. */
  u = 0;
  for(i=0;i<k;i++)
    if(icnst[i].deg == 0)
      for(j=0;j<icnst[i].nz;j++)
        if(icnst[i].idx[j] > u) u = icnst[i].idx[j];
  dbase = _malloc((u+1)*sizeof(polx*));
  cov = _malloc(MAX(nb,1)*sizeof(phirun));
  phi = (polx(*)[n])ocnst->phi;
  v = 0;
  for(i=0;i<=u;i++) {
    dbase[i] = &(*phi)[v];
    phi += pi->nu[i];
    v = (pi->nu[i]) ? 0 : v + pi->n[i];
  }

  nb = 0;
  m = 0;  // deg 0 constraints get consecutive challenges
  for(i=0;i<k;i++) {
    if(icnst[i].deg) continue;

    s = 0;
    for(j=0;j<QBYTES;j++)
      s |= (int64_t)hashbuf[16+m*QBYTES+j] << 8*j;
    s &= ((int64_t)1 << LOGQ) - 1;
    m += 1;

    if(icnst[i].b)
      polx_scale_add(ocnst->b,icnst[i].b,s);

    for(j=0;j<icnst[i].nz;j++) {
      polx *dst = dbase[icnst[i].idx[j]] + icnst[i].off[j];
      polxvec_scale_add(dst,icnst[i].phi[j],icnst[i].len[j],s);
      cov[nb].p = dst;
      cov[nb].len = icnst[i].len[j];
      nb += 1;
    }

    sparsemat_scale_add(ocnst->a,icnst[i].a,s);
  }

  /* Scaling by a full width challenge blows the CRT representatives up to ~q^2.
   * They are multiplied by the aggregation challenge, accumulated over LIFTS and
   * finally multiplied by the witness in amortize(), which overflows the product
   * of the polx primes and silently corrupts the value mod q. Reduce mod q here,
   * where the growth happens; prover and verifier both call this. Only the runs
   * that were scaled grew; the rest of phi is what jlproj_collapsmat left there,
   * already reduced, so the merged coverage is enough. */
  qsort(cov,nb,sizeof(phirun),phirun_cmp);
  for(i=0;i<nb;) {
    polx *lo = cov[i].p;
    polx *hi = lo + cov[i].len;
    for(j=i+1;j<nb && cov[j].p <= hi;j++)
      if(cov[j].p + cov[j].len > hi) hi = cov[j].p + cov[j].len;
    nc = hi - lo;
    polxvec_refresh(lo,nc);
    i = j;
  }

  free(cov);
  free(dbase);
}

/* One (destination, phi buffer, challenge) contribution to the aggregated phi. */
typedef struct {
  polx *dst;
  const polx *src;
  const polx *alpha;
  size_t len;
} aggent;

static int aggent_cmp(const void *x, const void *y) {
  const polx *const a = ((const aggent*)x)->src;
  const polx *const b = ((const aggent*)y)->src;
  return (a > b) - (a < b);
}

void aggregate_sparsecnst(statement *ost, const proof *pi, const sparsecnst *cnst, size_t k) {
  size_t i,j,u,v,ne,degtot,maxidx;
  __attribute__((aligned(16)))
  uint8_t hashbuf[32];
  polx (*phi)[ost->n];
  polx **dbase;
  polx *alphabuf;
  aggent *ent;

  shake128(hashbuf,32,ost->h,16);
  memcpy(ost->h,hashbuf,16);

  /* Destination base of every witness index, walked once instead of once per block. */
  maxidx = 0;
  degtot = 0;
  ne = 0;
  for(i=0;i<k;i++) {
    if(!cnst[i].deg) continue;
    degtot += cnst[i].deg;
    ne += cnst[i].nz;
    for(j=0;j<cnst[i].nz;j++)
      if(cnst[i].idx[j] > maxidx) maxidx = cnst[i].idx[j];
  }
  dbase = _malloc((maxidx+1)*sizeof(polx*));
  phi = (polx(*)[ost->n])ost->cnst->phi;
  v = 0;
  for(u=0;u<=maxidx;u++) {
    dbase[u] = &(*phi)[v];
    phi += pi->nu[u];
    v = (pi->nu[u]) ? 0 : v + pi->n[u];
  }

  alphabuf = _aligned_alloc(64,MAX(degtot,1)*sizeof(polx));
  ent = _malloc(MAX(ne,1)*sizeof(aggent));

  /* The challenges, b and a are consumed in the original order; only the phi
   * accumulation is deferred and regrouped. */
  ne = 0;
  degtot = 0;
  for(i=0;i<k;i++) {
    if(cnst[i].deg == 0) continue;
    polx *alpha = &alphabuf[degtot];
    degtot += cnst[i].deg;
    polxvec_quarternary(alpha,cnst[i].deg,&hashbuf[16],i);  // nonce is the global index

    if(cnst[i].b)
      polxvec_sprod_add(ost->cnst->b,alpha,cnst[i].b,cnst[i].deg);

    for(j=0;j<cnst[i].nz;j++) {
      polx *dst = dbase[cnst[i].idx[j]] + cnst[i].off[j];
      if(cnst[i].deg == 1 && cnst[i].mult[j] == 1) {
        ent[ne].dst = dst;
        ent[ne].src = cnst[i].phi[j];
        ent[ne].alpha = alpha;
        ent[ne].len = cnst[i].len[j];
        ne += 1;
      }
      else
        polxvec_collaps_add_extension(dst,alpha,cnst[i].phi[j],cnst[i].len[j],
                                      cnst[i].deg/cnst[i].mult[j],cnst[i].mult[j]);
    }

    sparsemat_polx_mul_add(ost->cnst->a,alpha,cnst[i].a);
  }

  /* Constraints alias each other's phi buffers but write few distinct destination
   * runs; visiting the contributions destination-major keeps the accumulator hot
   * instead of streaming it once per constraint. */
  qsort(ent,ne,sizeof(aggent),aggent_cmp);
  for(i=0;i<ne;) {
    size_t e,off,maxlen = ent[i].len;
    for(j=i+1;j<ne && ent[j].src == ent[i].src;j++)
      if(ent[j].len > maxlen) maxlen = ent[j].len;
    /* Tiled so the destination slice stays in L1 while every contribution to it
     * is streamed in. */
    for(off=0;off<maxlen;off+=AGGCHUNK)
      for(e=i;e<j;e++)
        if(off < ent[e].len)
          polxvec_polx_mul_add(ent[e].dst+off,ent[e].alpha,ent[e].src+off,
                               MIN(AGGCHUNK,ent[e].len-off));
    i = j;
  }

  free(ent);
  free(alphabuf);
  free(dbase);

  ost->cnst->a->coeffs = realloc(ost->cnst->a->coeffs,ost->cnst->a->len*sizeof(polx));
}

int principle_prove(statement *ost, witness *owt, proof *pi, const prncplstmnt *ist, const witness *iwt, int tail) {
  int ret;
  size_t i;
  constraint cnst[1] = {};
  void *buf = NULL;

  ret = init_proof(pi,iwt,ist->quadratic,tail);
  if(ret)  // commitments not secure (1/2)
    return ret;
  init_statement(ost,pi,ist->h);
  init_witness(owt,ost);
  printf("Predicted witness norm: %.2f\n\n",sqrt(pi->normsq));

  {
    buf = _aligned_alloc(64,ost->r*ost->n*(sizeof(polx)+256*N/8));
    polx (*sx)[ost->n] = (polx(*)[ost->n])buf;
    uint8_t (*jlmat)[ost->n][256*N/8] = (uint8_t(*)[ost->n][256*N/8])sx[ost->r];
    commit(ost,owt,pi,sx,iwt);
    ret = project(ost,pi,jlmat,iwt);
    if(ret) {
      ret += 10;
      goto err;
    }

    init_constraint(cnst,ost);
    for(i=0;i<LIFTS;i++) {
      collaps_jlproj(cnst,ost,pi,jlmat);
      collaps_sparsecnst(cnst,ost,pi,ist->cnst,ist->k);
      lift_aggregate_zqcnst(ost,pi,i,cnst,sx);
    }
    free_constraint(cnst);

    aggregate_sparsecnst(ost,pi,ist->cnst,ist->k);
    amortize(ost,owt,pi,sx);
    free(buf);
    buf = NULL;
  }

  polx_refresh(ost->cnst->b);
  polxvec_refresh(ost->cnst->phi,ost->n);
  return 0;

err:
  free(buf);
  free_constraint(cnst);
  free_proof(pi);
  free_statement(ost);
  free_witness(owt);
  return ret;
}

int principle_reduce(statement *ost, const proof *pi, const prncplstmnt *ist) {
  size_t i;
  int ret;
  uint8_t (*jlmat)[ost->n][256*N/8];
  constraint cnst[1] = {};

  init_statement(ost,pi,ist->h);
  jlmat = _aligned_alloc(64,ost->r*ost->n*256*N/8);

  reduce_commit(ost,pi);
  ret = reduce_project(ost,jlmat,pi,pi->r,ist->betasq);
  if(ret) goto err;  // projection too long

  init_constraint(cnst,ost);
  for(i=0;i<LIFTS;i++) {
    collaps_jlproj(cnst,ost,pi,jlmat);
    collaps_sparsecnst(cnst,ost,pi,ist->cnst,ist->k);
    reduce_lift_aggregate_zqcnst(ost,pi,i,cnst);
  }
  free_constraint(cnst);
  free(jlmat);
  jlmat = NULL;

  aggregate_sparsecnst(ost,pi,ist->cnst,ist->k);
  ret = reduce_amortize(ost,pi);
  if(ret) {  // commitments not secure (1/2)
    ret += 10;
    goto err;
  }

  polx_refresh(ost->cnst->b);
  polxvec_refresh(ost->cnst->phi,ost->n);
  return 0;

err:
  free_statement(ost);
  free(jlmat);
  free_constraint(cnst);
  return ret;
}

int principle_verify(const prncplstmnt *st, const witness *wt) {
  int ret;
  size_t i;
  uint64_t normsq = 0;
  polx *sx[wt->r];

  if(wt->r != st->r) {
    fprintf(stderr,"ERROR in principle_verify(): Mismatch in witness multiplicity\n");
    return 1;
  }
  for(i=0;i<wt->r;i++)
    if(wt->n[i] != st->n[i]) {
      fprintf(stderr,"ERROR in principle_verify(): Mismatch in length of witness vector %zu\n",i);
      return 2;
    }
  for(i=0;i<wt->r;i++)
    normsq += polyvec_sprodz(wt->s[i],wt->s[i],wt->n[i]);
  if(normsq > st->betasq) {
    fprintf(stderr,"ERROR in principle_verify(): Total witness vector norm too big\n");
    return 3;
  }

  *sx = NULL;
  for(i=0;i<st->k;i++) {
    ret = !sparsecnst_check(&st->cnst[i],sx,wt);
    if(ret) {
      fprintf(stderr,"ERROR in principle_verify(): Sparse dot-product constraint %zu does not hold\n",i);
      ret = 10+i;
      goto end;
    }
  }

end:
  free(*sx);
  return ret;
}
