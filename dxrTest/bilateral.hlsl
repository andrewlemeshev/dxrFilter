#define PI 3.14159265358979323846264338327950288419716939937510
#define EPS 1e-5
#define EQUAL_EPSILON 0.0001f

static const uint diameter = 31;
//static const float sigmaI = 12.0f;
//static const float sigmaS = 16.0f;

static const float sigmaS = 6.0f;
static const float sigmaL = 4.0f;

static const uint2 resolution = uint2(1280, 720);

//RWTexture2D<float4> output : register(u0);
//Texture2D<float4> colors : register(t0);
RWTexture2D<float2> output : register(u0);
Texture2D<float2> colors : register(t0);
Texture2D<float4> normals : register(t1);
Texture2D<uint2> pixelDatas : register(t2);

float distance(const uint2 xy, const uint2 ij) {
  const int xi = int(xy.x - ij.x);
  const int yj = int(xy.y - ij.y);
  return float(sqrt(xi*xi + yj* yj));
}

float gaussian(const float x, const float sigma) {
  return exp(-(x*x / (2 * sigma*sigma)) / (2 * PI * sigma*sigma));
}

float lum(const float4 color) {
  return length(color.xyz);
}

bool eq(const float4 first, const float4 second) {
  const float4 tmp = abs(first - second);

  return tmp.x < EQUAL_EPSILON && tmp.y < EQUAL_EPSILON && tmp.z < EQUAL_EPSILON && tmp.w < EQUAL_EPSILON;
}

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
  const uint2 coord = uint2(DTid.x, DTid.y);

  const float sigS = max(sigmaS, EPS);
  const float sigL = max(sigmaL, EPS);

  const float facS = -1. / (2.*sigS*sigS);
  const float facL = -1. / (2.*sigL*sigL);

  float sumW = 0.0f;
  //float4 sumC = float4(0.0f, 0.0f, 0.0f, 0.0f);
  float sumC = 0.0f;
  const float halfSize = sigS * 2;
  const uint radius = diameter / 2;

  //const float l = lum(colors[coord]);
  const float2 current = colors[coord];
  const float l = current.x;
  const float dist = current.y;
  const float4 normal = normals[coord];
  const uint2 pixelData = pixelDatas[coord];
  const float coef = float(pixelData.y) / float(pixelData.x);

  // мне нужно что-то сделать с диаметром 
  // как то его нужно менять в зависимости от того что у меня есть
  // у меня есть дальность до объекта (как ее правильно оценить только?)
  // у меня есть среднее число пикселей в округе
  // если число =1 то по идее ничего делать не надо (то есть 0)
  // если число =0 то что делать? тут не всегда очевидно (но по идее если увеличить диаметр на прошлом шаге, то будет точнее)
  // если число от 0 до 1 как действовать? думаю что в этом случае нужно пользоваться другими данными
  // rayHitDist легко можно оценить по дальности до источника освещения, дает ли это мне чтото конкретное?
  // сомневаюсь, мне по идее как минимум нужен масимальный хитдистанс у каждого объекта
  // чет пока не понимаю как тут подступиться, предположим у нас есть масимальный хитдистанс
  // чем дальше хит дист тем больше размывать, но скорее всего у разных точек разный вес
  // + ко всему прочему не у всех точек есть этот райхитдист

  // это более менее помогает, но остаются белые точки
  // теперь белых точек меньше, но все равно неочень
  const uint newDiameter = uint(pixelData.y != 0 || pixelData.y != pixelData.x) * diameter * max(abs(l - coef), 0.2f); // 1.0f - coef

  // тогда сильно размывается тень, из-за чего появляются проблемы на краях
  //const uint newDiameter = uint(pixelData.y != 0) * diameter;
  for (uint i = 0; i < newDiameter; i++) {
    for (uint j = 0; j < newDiameter; j++) {
      const int2 offset = int2(radius - i, radius - j);
      const uint2 neighbor = coord + offset;

      const float4 normalNeighbor = normals[neighbor];
      const uint2 neighborPixelData = pixelDatas[neighbor];
      const float neighborCoef = float(neighborPixelData.y) / float(neighborPixelData.x);
      //if (!eq(normal, normalNeighbor)) continue;
      const bool b = eq(normal, normalNeighbor);

      const float2 offsetData = colors[neighbor];
      const float offsetColor = offsetData.x;
      const float dist = offsetData.y;

      // короч, мне нужно как то добавить сюда данные о дальности
      // причем так чтобы эта дальность увеличивала вес этого пикселя
      // чем больше дальность тем больше вес? вообще это по идее не важно
      // так как мне нужно чтобы пиксель в котором тень есть давал чуть больше,
      // чем пиксель в котором тени нет
      // как это сделать? насколько больше?
      
      // умножение размывает чуть больше пенумбру которая находится у нормальной тени
      // это более менее то что нужно, но я не понимаю до конца зависимость

      // суть такова, мне нужно чтобы тень "затухала" по мере удаления от основной тени
      // помоему в реальной жизни тень "затухает" не линейно, но мне бы хотя бы и такое поведение
      // как это сделать? как раз таки я думал что с этим поможет пересчитать пиксели вокруг
      // то есть в квадрате 9х9 есть например 1 пиксель с тенью

      // короч, считаю количество пикселей
      // на картинке получается более менее затухающее изображение с некоторыми артефактами
      // разница между верхним и нижними значениями сильно уменьшилась
      // меня лишь беспокоят артефакты на самых краях, 
      // но с другой стороны нам должен помочь rayHitDist 
      // я вообще могу просто откидывать ситуации в которых там меньше 10% попаданий
      // но скорее всего мне нужно будет это делать по умному, 
      // например мало процентов и большой хит - откидываем
      // что делать с остальным? 
      // вообще билатеральный фильтр по этим значениям может сделать все даже лучше чем с расстояниями

      const float distS = length(float2(offset)) * dist * (0.5f);
      //float distL = lum(offsetColor) - l;
      //const float distL = offsetColor - l;
      const float distL = neighborCoef - coef;

      const float wS = exp(facS*float(distS*distS));
      const float wL = exp(facL*float(distL*distL));
      const float w = wS * wL * float(b);

      sumW += w;
      sumC += offsetColor * w;
    }
  }

  output[coord] = sumW == 0.0f ? float2(l, dist) : float2(sumC / sumW, dist);
  //output[coord] = colors[coord];
}