#include "utils.h"

typedef enum {
  TEXTURE_TYPE_NONE,  // default
  TEXTURE_TYPE_SPHERE,
  TEXTURE_TYPE_CUBE_TOP,
  TEXTURE_TYPE_CUBE_BOTTOM,
  TEXTURE_TYPE_CUBE_FRONT,
  TEXTURE_TYPE_CUBE_BACK,
  TEXTURE_TYPE_CUBE_LEFT,
  TEXTURE_TYPE_CUBE_RIGHT
} texture_type_t;

struct texture_option_t {
  texture_type_t type;      // -type (default TEXTURE_TYPE_NONE)
  real_t sharpness;         // -boost (default 1.0?)
  real_t brightness;        // base_value in -mm option (default 0)
  real_t contrast;          // gain_value in -mm option (default 1)
  real_t origin_offset[3];  // -o u [v [w]] (default 0 0 0)
  real_t scale[3];          // -s u [v [w]] (default 1 1 1)
  real_t turbulence[3];     // -t u [v [w]] (default 0 0 0)
  int texture_resolution;   // -texres resolution (No default value in the spec.
                            // We'll use -1)
  bool clamp;               // -clamp (default false)
  char imfchan;  // -imfchan (the default for bump is 'l' and for decal is 'm')
  bool blendu;   // -blendu (default on)
  bool blendv;   // -blendv (default on)
  real_t bump_multiplier;  // -bm (for bump maps only, default 1.0)

  // extension
  std::string colorspace;  // Explicitly specify color space of stored texel
                           // value. Usually `sRGB` or `linear` (default empty).
};

struct material_t {
  std::string name;

  real_t ambient[3];
  real_t diffuse[3];
  real_t specular[3];
  real_t transmittance[3];
  real_t emission[3];
  real_t shininess;
  real_t ior;       // index of refraction
  real_t dissolve;  // 1 == opaque; 0 == fully transparent
  // illumination model (see http://www.fileformat.info/format/material/)
  int illum;

  int dummy;  // Suppress padding warning.

  std::string ambient_texname;             // map_Ka
  std::string diffuse_texname;             // map_Kd
  std::string specular_texname;            // map_Ks
  std::string specular_highlight_texname;  // map_Ns
  std::string bump_texname;                // map_bump, map_Bump, bump
  std::string displacement_texname;        // disp
  std::string alpha_texname;               // map_d
  std::string reflection_texname;          // refl

  texture_option_t ambient_texopt;
  texture_option_t diffuse_texopt;
  texture_option_t specular_texopt;
  texture_option_t specular_highlight_texopt;
  texture_option_t bump_texopt;
  texture_option_t displacement_texopt;
  texture_option_t alpha_texopt;
  texture_option_t reflection_texopt;

  // PBR extension
  // http://exocortex.com/blog/extending_wavefront_mtl_to_support_pbr
  real_t roughness;            // [0, 1] default 0
  real_t metallic;             // [0, 1] default 0
  real_t sheen;                // [0, 1] default 0
  real_t clearcoat_thickness;  // [0, 1] default 0
  real_t clearcoat_roughness;  // [0, 1] default 0
  real_t anisotropy;           // aniso. [0, 1] default 0
  real_t anisotropy_rotation;  // anisor. [0, 1] default 0
  real_t pad0;
  std::string roughness_texname;  // map_Pr
  std::string metallic_texname;   // map_Pm
  std::string sheen_texname;      // map_Ps
  std::string emissive_texname;   // map_Ke
  std::string normal_texname;     // norm. For normal mapping.

  texture_option_t roughness_texopt;
  texture_option_t metallic_texopt;
  texture_option_t sheen_texopt;
  texture_option_t emissive_texopt;
  texture_option_t normal_texopt;

  int pad2;

  std::map<std::string, std::string> unknown_parameter;

  #ifdef TINY_OBJ_LOADER_PYTHON_BINDING
    // For pybind11
    std::array<double, 3> GetDiffuse() {
      std::array<double, 3> values;
      values[0] = double(diffuse[0]);
      values[1] = double(diffuse[1]);
      values[2] = double(diffuse[2]);

      return values;
    }

    std::array<double, 3> GetSpecular() {
      std::array<double, 3> values;
      values[0] = double(specular[0]);
      values[1] = double(specular[1]);
      values[2] = double(specular[2]);

      return values;
    }

    std::array<double, 3> GetTransmittance() {
      std::array<double, 3> values;
      values[0] = double(transmittance[0]);
      values[1] = double(transmittance[1]);
      values[2] = double(transmittance[2]);

      return values;
    }

    std::array<double, 3> GetEmission() {
      std::array<double, 3> values;
      values[0] = double(emission[0]);
      values[1] = double(emission[1]);
      values[2] = double(emission[2]);

      return values;
    }

    std::array<double, 3> GetAmbient() {
      std::array<double, 3> values;
      values[0] = double(ambient[0]);
      values[1] = double(ambient[1]);
      values[2] = double(ambient[2]);

      return values;
    }

    void SetDiffuse(std::array<double, 3> &a) {
      diffuse[0] = real_t(a[0]);
      diffuse[1] = real_t(a[1]);
      diffuse[2] = real_t(a[2]);
    }

    void SetAmbient(std::array<double, 3> &a) {
      ambient[0] = real_t(a[0]);
      ambient[1] = real_t(a[1]);
      ambient[2] = real_t(a[2]);
    }

    void SetSpecular(std::array<double, 3> &a) {
      specular[0] = real_t(a[0]);
      specular[1] = real_t(a[1]);
      specular[2] = real_t(a[2]);
    }

    void SetTransmittance(std::array<double, 3> &a) {
      transmittance[0] = real_t(a[0]);
      transmittance[1] = real_t(a[1]);
      transmittance[2] = real_t(a[2]);
    }

    std::string GetCustomParameter(const std::string &key) {
      std::map<std::string, std::string>::const_iterator it =
          unknown_parameter.find(key);

      if (it != unknown_parameter.end()) {
        return it->second;
      }
      return std::string();
    }

  #endif
  };

static inline texture_type_t parseTextureType(
					      const char **token, texture_type_t default_value = TEXTURE_TYPE_NONE) {
  (*token) += strspn((*token), " \t");
  const char *end = (*token) + strcspn((*token), " \t\r");
  texture_type_t ty = default_value;

  if ((0 == strncmp((*token), "cube_top", strlen("cube_top")))) {
    ty = TEXTURE_TYPE_CUBE_TOP;
  } else if ((0 == strncmp((*token), "cube_bottom", strlen("cube_bottom")))) {
    ty = TEXTURE_TYPE_CUBE_BOTTOM;
  } else if ((0 == strncmp((*token), "cube_left", strlen("cube_left")))) {
    ty = TEXTURE_TYPE_CUBE_LEFT;
  } else if ((0 == strncmp((*token), "cube_right", strlen("cube_right")))) {
    ty = TEXTURE_TYPE_CUBE_RIGHT;
  } else if ((0 == strncmp((*token), "cube_front", strlen("cube_front")))) {
    ty = TEXTURE_TYPE_CUBE_FRONT;
  } else if ((0 == strncmp((*token), "cube_back", strlen("cube_back")))) {
    ty = TEXTURE_TYPE_CUBE_BACK;
  } else if ((0 == strncmp((*token), "sphere", strlen("sphere")))) {
    ty = TEXTURE_TYPE_SPHERE;
  }

  (*token) = end;
  return ty;
}

bool ParseTextureNameAndOption(std::string *texname, texture_option_t *texopt,
                               const char *linebuf) {
  // @todo { write more robust lexer and parser. }
  bool found_texname = false;
  std::string texture_name;

  const char *token = linebuf;  // Assume line ends with NULL

  while (!IS_NEW_LINE((*token))) {
    token += strspn(token, " \t");  // skip space
    if ((0 == strncmp(token, "-blendu", 7)) && IS_SPACE((token[7]))) {
      token += 8;
      texopt->blendu = parseOnOff(&token, /* default */ true);
    } else if ((0 == strncmp(token, "-blendv", 7)) && IS_SPACE((token[7]))) {
      token += 8;
      texopt->blendv = parseOnOff(&token, /* default */ true);
    } else if ((0 == strncmp(token, "-clamp", 6)) && IS_SPACE((token[6]))) {
      token += 7;
      texopt->clamp = parseOnOff(&token, /* default */ true);
    } else if ((0 == strncmp(token, "-boost", 6)) && IS_SPACE((token[6]))) {
      token += 7;
      texopt->sharpness = parseReal(&token, 1.0);
    } else if ((0 == strncmp(token, "-bm", 3)) && IS_SPACE((token[3]))) {
      token += 4;
      texopt->bump_multiplier = parseReal(&token, 1.0);
    } else if ((0 == strncmp(token, "-o", 2)) && IS_SPACE((token[2]))) {
      token += 3;
      parseReal3(&(texopt->origin_offset[0]), &(texopt->origin_offset[1]),
                 &(texopt->origin_offset[2]), &token);
    } else if ((0 == strncmp(token, "-s", 2)) && IS_SPACE((token[2]))) {
      token += 3;
      parseReal3(&(texopt->scale[0]), &(texopt->scale[1]), &(texopt->scale[2]),
                 &token, 1.0, 1.0, 1.0);
    } else if ((0 == strncmp(token, "-t", 2)) && IS_SPACE((token[2]))) {
      token += 3;
      parseReal3(&(texopt->turbulence[0]), &(texopt->turbulence[1]),
                 &(texopt->turbulence[2]), &token);
    } else if ((0 == strncmp(token, "-type", 5)) && IS_SPACE((token[5]))) {
      token += 5;
      texopt->type = parseTextureType((&token), TEXTURE_TYPE_NONE);
    } else if ((0 == strncmp(token, "-texres", 7)) && IS_SPACE((token[7]))) {
      token += 7;
      // TODO(syoyo): Check if arg is int type.
      texopt->texture_resolution = parseInt(&token);
    } else if ((0 == strncmp(token, "-imfchan", 8)) && IS_SPACE((token[8]))) {
      token += 9;
      token += strspn(token, " \t");
      const char *end = token + strcspn(token, " \t\r");
      if ((end - token) == 1) {  // Assume one char for -imfchan
        texopt->imfchan = (*token);
      }
      token = end;
    } else if ((0 == strncmp(token, "-mm", 3)) && IS_SPACE((token[3]))) {
      token += 4;
      parseReal2(&(texopt->brightness), &(texopt->contrast), &token, 0.0, 1.0);
    } else if ((0 == strncmp(token, "-colorspace", 11)) &&
               IS_SPACE((token[11]))) {
      token += 12;
      texopt->colorspace = parseString(&token);
    } else {
      // Assume texture filename
#if 0
      size_t len = strcspn(token, " \t\r");  // untile next space
      texture_name = std::string(token, token + len);
      token += len;

      token += strspn(token, " \t");  // skip space
#else
      // Read filename until line end to parse filename containing whitespace
      // TODO(syoyo): Support parsing texture option flag after the filename.
      texture_name = std::string(token);
      token += texture_name.length();
#endif

      found_texname = true;
    }
  }

  if (found_texname) {
    (*texname) = texture_name;
    return true;
  } else {
    return false;
  }
}

static void InitTexOpt(texture_option_t *texopt, const bool is_bump) {
  if (is_bump) {
    texopt->imfchan = 'l';
  } else {
    texopt->imfchan = 'm';
  }
  texopt->bump_multiplier = static_cast<real_t>(1.0);
  texopt->clamp = false;
  texopt->blendu = true;
  texopt->blendv = true;
  texopt->sharpness = static_cast<real_t>(1.0);
  texopt->brightness = static_cast<real_t>(0.0);
  texopt->contrast = static_cast<real_t>(1.0);
  texopt->origin_offset[0] = static_cast<real_t>(0.0);
  texopt->origin_offset[1] = static_cast<real_t>(0.0);
  texopt->origin_offset[2] = static_cast<real_t>(0.0);
  texopt->scale[0] = static_cast<real_t>(1.0);
  texopt->scale[1] = static_cast<real_t>(1.0);
  texopt->scale[2] = static_cast<real_t>(1.0);
  texopt->turbulence[0] = static_cast<real_t>(0.0);
  texopt->turbulence[1] = static_cast<real_t>(0.0);
  texopt->turbulence[2] = static_cast<real_t>(0.0);
  texopt->texture_resolution = -1;
  texopt->type = TEXTURE_TYPE_NONE;
}

static void InitMaterial(material_t *material) {
  InitTexOpt(&material->ambient_texopt, /* is_bump */ false);
  InitTexOpt(&material->diffuse_texopt, /* is_bump */ false);
  InitTexOpt(&material->specular_texopt, /* is_bump */ false);
  InitTexOpt(&material->specular_highlight_texopt, /* is_bump */ false);
  InitTexOpt(&material->bump_texopt, /* is_bump */ true);
  InitTexOpt(&material->displacement_texopt, /* is_bump */ false);
  InitTexOpt(&material->alpha_texopt, /* is_bump */ false);
  InitTexOpt(&material->reflection_texopt, /* is_bump */ false);
  InitTexOpt(&material->roughness_texopt, /* is_bump */ false);
  InitTexOpt(&material->metallic_texopt, /* is_bump */ false);
  InitTexOpt(&material->sheen_texopt, /* is_bump */ false);
  InitTexOpt(&material->emissive_texopt, /* is_bump */ false);
  InitTexOpt(&material->normal_texopt,
             /* is_bump */ false);  // @fixme { is_bump will be true? }
  material->name = "";
  material->ambient_texname = "";
  material->diffuse_texname = "";
  material->specular_texname = "";
  material->specular_highlight_texname = "";
  material->bump_texname = "";
  material->displacement_texname = "";
  material->reflection_texname = "";
  material->alpha_texname = "";
  for (int i = 0; i < 3; i++) {
    material->ambient[i] = static_cast<real_t>(0.0);
    material->diffuse[i] = static_cast<real_t>(0.0);
    material->specular[i] = static_cast<real_t>(0.0);
    material->transmittance[i] = static_cast<real_t>(0.0);
    material->emission[i] = static_cast<real_t>(0.0);
  }
  material->illum = 0;
  material->dissolve = static_cast<real_t>(1.0);
  material->shininess = static_cast<real_t>(1.0);
  material->ior = static_cast<real_t>(1.0);

  material->roughness = static_cast<real_t>(0.0);
  material->metallic = static_cast<real_t>(0.0);
  material->sheen = static_cast<real_t>(0.0);
  material->clearcoat_thickness = static_cast<real_t>(0.0);
  material->clearcoat_roughness = static_cast<real_t>(0.0);
  material->anisotropy_rotation = static_cast<real_t>(0.0);
  material->anisotropy = static_cast<real_t>(0.0);
  material->roughness_texname = "";
  material->metallic_texname = "";
  material->sheen_texname = "";
  material->emissive_texname = "";
  material->normal_texname = "";

  material->unknown_parameter.clear();
}

// code from https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
template <typename T>
static int pnpoly(int nvert, T *vertx, T *verty, T testx, T testy) {
  int i, j, c = 0;
  for (i = 0, j = nvert - 1; i < nvert; j = i++) {
    if (((verty[i] > testy) != (verty[j] > testy)) &&
        (testx <
         (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) +
	 vertx[i]))
      c = !c;
  }
  return c;
}

void LoadMtl(std::map<std::string, int> *material_map,
	      std::vector<material_t> *materials, std::istream *inStream,
	      std::string *warning, std::string *err) {
   (void)err;

   // Create a default material anyway.
   material_t material;
   InitMaterial(&material);

   // Issue 43. `d` wins against `Tr` since `Tr` is not in the MTL specification.
   bool has_d = false;
   bool has_tr = false;

   // has_kd is used to set a default diffuse value when map_Kd is present
   // and Kd is not.
   bool has_kd = false;

   std::stringstream warn_ss;

   size_t line_no = 0;
   std::string linebuf;
   while (inStream->peek() != -1) {
     safeGetline(*inStream, linebuf);
     line_no++;

     // Trim trailing whitespace.
     if (linebuf.size() > 0) {
       linebuf = linebuf.substr(0, linebuf.find_last_not_of(" \t") + 1);
     }

     // Trim newline '\r\n' or '\n'
     if (linebuf.size() > 0) {
       if (linebuf[linebuf.size() - 1] == '\n')
	 linebuf.erase(linebuf.size() - 1);
     }
     if (linebuf.size() > 0) {
       if (linebuf[linebuf.size() - 1] == '\r')
	 linebuf.erase(linebuf.size() - 1);
     }

     // Skip if empty line.
     if (linebuf.empty()) {
       continue;
     }

     // Skip leading space.
     const char *token = linebuf.c_str();
     token += strspn(token, " \t");

     assert(token);
     if (token[0] == '\0') continue;  // empty line

     if (token[0] == '#') continue;  // comment line

     // new mtl
     if ((0 == strncmp(token, "newmtl", 6)) && IS_SPACE((token[6]))) {
       // flush previous material.
       if (!material.name.empty()) {
	 material_map->insert(std::pair<std::string, int>(
							  material.name, static_cast<int>(materials->size())));
	 materials->push_back(material);
       }

       // initial temporary material
       InitMaterial(&material);

       has_d = false;
       has_tr = false;

       // set new mtl name
       token += 7;
       {
	 std::stringstream sstr;
	 sstr << token;
	 material.name = sstr.str();
       }
       continue;
     }

     // ambient
     if (token[0] == 'K' && token[1] == 'a' && IS_SPACE((token[2]))) {
       token += 2;
       real_t r, g, b;
       parseReal3(&r, &g, &b, &token);
       material.ambient[0] = r;
       material.ambient[1] = g;
       material.ambient[2] = b;
       continue;
     }

     // diffuse
     if (token[0] == 'K' && token[1] == 'd' && IS_SPACE((token[2]))) {
       token += 2;
       real_t r, g, b;
       parseReal3(&r, &g, &b, &token);
       material.diffuse[0] = r;
       material.diffuse[1] = g;
       material.diffuse[2] = b;
       has_kd = true;
       continue;
     }

     // specular
     if (token[0] == 'K' && token[1] == 's' && IS_SPACE((token[2]))) {
       token += 2;
       real_t r, g, b;
       parseReal3(&r, &g, &b, &token);
       material.specular[0] = r;
       material.specular[1] = g;
       material.specular[2] = b;
       continue;
     }

     // transmittance
     if ((token[0] == 'K' && token[1] == 't' && IS_SPACE((token[2]))) ||
	 (token[0] == 'T' && token[1] == 'f' && IS_SPACE((token[2])))) {
       token += 2;
       real_t r, g, b;
       parseReal3(&r, &g, &b, &token);
       material.transmittance[0] = r;
       material.transmittance[1] = g;
       material.transmittance[2] = b;
       continue;
     }

     // ior(index of refraction)
     if (token[0] == 'N' && token[1] == 'i' && IS_SPACE((token[2]))) {
       token += 2;
       material.ior = parseReal(&token);
       continue;
     }

     // emission
     if (token[0] == 'K' && token[1] == 'e' && IS_SPACE(token[2])) {
       token += 2;
       real_t r, g, b;
       parseReal3(&r, &g, &b, &token);
       material.emission[0] = r;
       material.emission[1] = g;
       material.emission[2] = b;
       continue;
     }

     // shininess
     if (token[0] == 'N' && token[1] == 's' && IS_SPACE(token[2])) {
       token += 2;
       material.shininess = parseReal(&token);
       continue;
     }

     // illum model
     if (0 == strncmp(token, "illum", 5) && IS_SPACE(token[5])) {
       token += 6;
       material.illum = parseInt(&token);
       continue;
     }

     // dissolve
     if ((token[0] == 'd' && IS_SPACE(token[1]))) {
       token += 1;
       material.dissolve = parseReal(&token);

       if (has_tr) {
        warn_ss << "Both `d` and `Tr` parameters defined for \""
                << material.name
                << "\". Use the value of `d` for dissolve (line " << line_no
                << " in .mtl.)\n";
       }
       has_d = true;
       continue;
     }
     if (token[0] == 'T' && token[1] == 'r' && IS_SPACE(token[2])) {
       token += 2;
       if (has_d) {
	 // `d` wins. Ignore `Tr` value.
        warn_ss << "Both `d` and `Tr` parameters defined for \""
                << material.name
                << "\". Use the value of `d` for dissolve (line " << line_no
                << " in .mtl.)\n";
       } else {
	 // We invert value of Tr(assume Tr is in range [0, 1])
	 // NOTE: Interpretation of Tr is application(exporter) dependent. For
	 // some application(e.g. 3ds max obj exporter), Tr = d(Issue 43)
	 material.dissolve = static_cast<real_t>(1.0) - parseReal(&token);
       }
       has_tr = true;
       continue;
     }

     // PBR: roughness
     if (token[0] == 'P' && token[1] == 'r' && IS_SPACE(token[2])) {
       token += 2;
       material.roughness = parseReal(&token);
       continue;
     }

     // PBR: metallic
     if (token[0] == 'P' && token[1] == 'm' && IS_SPACE(token[2])) {
       token += 2;
       material.metallic = parseReal(&token);
       continue;
     }

     // PBR: sheen
     if (token[0] == 'P' && token[1] == 's' && IS_SPACE(token[2])) {
       token += 2;
       material.sheen = parseReal(&token);
       continue;
     }

     // PBR: clearcoat thickness
     if (token[0] == 'P' && token[1] == 'c' && IS_SPACE(token[2])) {
       token += 2;
       material.clearcoat_thickness = parseReal(&token);
       continue;
     }

     // PBR: clearcoat roughness
     if ((0 == strncmp(token, "Pcr", 3)) && IS_SPACE(token[3])) {
       token += 4;
       material.clearcoat_roughness = parseReal(&token);
       continue;
     }

     // PBR: anisotropy
     if ((0 == strncmp(token, "aniso", 5)) && IS_SPACE(token[5])) {
       token += 6;
       material.anisotropy = parseReal(&token);
       continue;
     }

     // PBR: anisotropy rotation
     if ((0 == strncmp(token, "anisor", 6)) && IS_SPACE(token[6])) {
       token += 7;
       material.anisotropy_rotation = parseReal(&token);
       continue;
     }

     // ambient texture
     if ((0 == strncmp(token, "map_Ka", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.ambient_texname),
				 &(material.ambient_texopt), token);
       continue;
     }

     // diffuse texture
     if ((0 == strncmp(token, "map_Kd", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.diffuse_texname),
				 &(material.diffuse_texopt), token);

       // Set a decent diffuse default value if a diffuse texture is specified
       // without a matching Kd value.
       if (!has_kd) {
	 material.diffuse[0] = static_cast<real_t>(0.6);
	 material.diffuse[1] = static_cast<real_t>(0.6);
	 material.diffuse[2] = static_cast<real_t>(0.6);
       }

       continue;
     }

     // specular texture
     if ((0 == strncmp(token, "map_Ks", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.specular_texname),
				 &(material.specular_texopt), token);
       continue;
     }

     // specular highlight texture
     if ((0 == strncmp(token, "map_Ns", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.specular_highlight_texname),
				 &(material.specular_highlight_texopt), token);
       continue;
     }

     // bump texture
     if ((0 == strncmp(token, "map_bump", 8)) && IS_SPACE(token[8])) {
       token += 9;
       ParseTextureNameAndOption(&(material.bump_texname),
				 &(material.bump_texopt), token);
       continue;
     }

     // bump texture
     if ((0 == strncmp(token, "map_Bump", 8)) && IS_SPACE(token[8])) {
       token += 9;
       ParseTextureNameAndOption(&(material.bump_texname),
				 &(material.bump_texopt), token);
       continue;
     }

     // bump texture
     if ((0 == strncmp(token, "bump", 4)) && IS_SPACE(token[4])) {
       token += 5;
       ParseTextureNameAndOption(&(material.bump_texname),
				 &(material.bump_texopt), token);
       continue;
     }

     // alpha texture
     if ((0 == strncmp(token, "map_d", 5)) && IS_SPACE(token[5])) {
       token += 6;
       material.alpha_texname = token;
       ParseTextureNameAndOption(&(material.alpha_texname),
				 &(material.alpha_texopt), token);
       continue;
     }

     // displacement texture
     if ((0 == strncmp(token, "disp", 4)) && IS_SPACE(token[4])) {
       token += 5;
       ParseTextureNameAndOption(&(material.displacement_texname),
				 &(material.displacement_texopt), token);
       continue;
     }

     // reflection map
     if ((0 == strncmp(token, "refl", 4)) && IS_SPACE(token[4])) {
       token += 5;
       ParseTextureNameAndOption(&(material.reflection_texname),
				 &(material.reflection_texopt), token);
       continue;
     }

     // PBR: roughness texture
     if ((0 == strncmp(token, "map_Pr", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.roughness_texname),
				 &(material.roughness_texopt), token);
       continue;
     }

     // PBR: metallic texture
     if ((0 == strncmp(token, "map_Pm", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.metallic_texname),
				 &(material.metallic_texopt), token);
       continue;
     }

     // PBR: sheen texture
     if ((0 == strncmp(token, "map_Ps", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.sheen_texname),
				 &(material.sheen_texopt), token);
       continue;
     }

     // PBR: emissive texture
     if ((0 == strncmp(token, "map_Ke", 6)) && IS_SPACE(token[6])) {
       token += 7;
       ParseTextureNameAndOption(&(material.emissive_texname),
				 &(material.emissive_texopt), token);
       continue;
     }

     // PBR: normal map texture
     if ((0 == strncmp(token, "norm", 4)) && IS_SPACE(token[4])) {
       token += 5;
       ParseTextureNameAndOption(&(material.normal_texname),
				 &(material.normal_texopt), token);
       continue;
     }

     // unknown parameter
     const char *_space = strchr(token, ' ');
     if (!_space) {
       _space = strchr(token, '\t');
     }
     if (_space) {
       std::ptrdiff_t len = _space - token;
       std::string key(token, static_cast<size_t>(len));
       std::string value = _space + 1;
       material.unknown_parameter.insert(
					 std::pair<std::string, std::string>(key, value));
     }
   }
   // flush last material.
   material_map->insert(std::pair<std::string, int>(
						    material.name, static_cast<int>(materials->size())));
   materials->push_back(material);

   if (warning) {
     (*warning) = warn_ss.str();
   }
 }

class MaterialReader {
 public:
  MaterialReader() {}

  virtual bool operator()(const std::string &matId,
			  std::vector<material_t> *materials,
			  std::map<std::string, int> *matMap, std::string *warn,
			  std::string *err) = 0;
};

class MaterialStreamReader : public MaterialReader {
 public:
  explicit MaterialStreamReader(std::istream &inStream)
    : m_inStream(inStream) {}

  bool operator()(const std::string &matId,
				       std::vector<material_t> *materials,
				       std::map<std::string, int> *matMap,
				       std::string *warn, std::string *err) {
   (void)err;
   (void)matId;
   if (!m_inStream) {
     std::stringstream ss;
     ss << "Material stream in error state. \n";
     if (warn) {
       (*warn) += ss.str();
     }
     return false;
   }

   LoadMtl(matMap, materials, &m_inStream, warn, err);

   return true;
 }

 private:
 std::istream &m_inStream;

};
