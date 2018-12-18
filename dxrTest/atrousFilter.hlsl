// входные данные: рт изображение, нормали, глубина
// выходные: фильтрованное изображение

// + на вход подаются какие то числа, я еще не понял какие

// https://jo.dreggn.org/home/2010_atrous.pdf
// попытаюсь сделать это

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {

}

//[numthreads(256, 1, 1)]
//void main(uint3 DTid : SV_DispatchThreadID) {
//  float4 sum = float4(0.0);
//  float2 step = float2(1. / 512., 1. / 512.);
//  // resolution
//  float4 cval = texture2D(colorMap, gl_TexCoord[0].st);
//  float4 nval = texture2D(normalMap, gl_TexCoord[0].st);
//  float4 pval = texture2D(posMap, gl_TexCoord[0].st);
//  float cum_w = 0.0;
//  for (uint i = 0; i < 25; ++i) {
//    float2 uv = gl_TexCoord[0].st + offset[i] * step * stepwidth;
//    float4 ctmp = texture2D(colorMap, uv);
//    float4 t = cval - ctmp;
//
//    float dist2 = dot(t, t);
//    float c_w = min(exp(-(dist2) / c_phi), 1.0);
//
//    float4 ntmp = texture2D(normalMap, uv);
//    t = nval - ntmp;
//    dist2 = max(dot(t, t) / (stepwidth * stepwidth), 0.0);
//
//    float n_w = min(exp(-(dist2) / n_phi), 1.0);
//    float4 ptmp = texture2D(posMap, uv);
//    t = pval - ptmp;
//    dist2 = dot(t, t);
//
//    float p_w = min(exp(-(dist2) / p_phi), 1.0);
//    float weight = c_w * n_w * p_w;
//
//    sum += ctmp * weight * kernel[i];
//    cum_w += weight * kernel[i];
//  }
//  gl_FragData[0] = sum / cum_w;
//}