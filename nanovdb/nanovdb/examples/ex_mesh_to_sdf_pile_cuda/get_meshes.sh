#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# get_meshes.sh [directory=meshes]: downloads the 24 OBJ meshes of the scene (222 MB) from
# https://github.com/odedstein/meshes at a fixed commit and checks their SHA-256 sums. A file that is already there
# with the right sum is not downloaded again. Each mesh has its own author and license: see README.md.
set -u
DIR=${1:-meshes}
COMMIT=835213277571d625722d25880c9c3010c6f45a5a
mkdir -p "$DIR"
STATUS=0
while read -r SUM OBJECT FILE; do
  if ! echo "$SUM  $DIR/$FILE" | sha256sum --check --status 2> /dev/null; then
    curl -L --fail --silent --show-error -o "$DIR/$FILE" \
      "https://raw.githubusercontent.com/odedstein/meshes/$COMMIT/objects/$OBJECT/$FILE" || STATUS=1
  fi
  if echo "$SUM  $DIR/$FILE" | sha256sum --check --status 2> /dev/null; then echo "ok      $FILE"; else echo "FAILED  $FILE"; STATUS=1; fi
done <<'LIST'
a331abaf7dbaae6ad10045f244bb372c4735d4c0da43054abf936cb00d2f84ea nefertiti nefertiti.obj
1a987b204e5599a8ceb640d7be8d2a8288c5b50fbd9d7792b84d0d6df1f9a435 cat cat.obj
fb5c4347a67870fe064a71c637515ee0b689dd010aef677a154eb2752026ac39 cheese cheese.obj
c23a72e4a4c0182847d1ef7f2808426e8ba5ced3514fac25869b0e9211df88c3 tree tree_closed.obj
0e6a01c7cfb52a59d31c01af7e1dfea9d43ecee0684c899716392d7470fa6145 violin violin.obj
0f32b65d098bcc5b4e79734d3cac7ea2d7a888c1c477235e0fedf8fcdb6d01a0 koala koala.obj
823d83ed28a41ba714163bec7922c6bb7a82e9d82810ced1cc675660309e1817 skull skull.obj
98370d678a48e32191920f485725aa6be729f4a9f06f3cab9a8853a23a374384 scorpion scorpion.obj
6afd245b8cf79768c4e7eca5b960119a476dc0a857f978d73dda94eb1279b723 hammer hammer.obj
966866fd198ba81f055f6fe4c03278f1b4991fee793317ad9553c07435fb2961 bunny bunny_hr.obj
83cffb5170f69c094d6b2b8421de4730555784d25e2373dadecaaadb4b2a4abd sword sword.obj
d2337237da8cabc6584e1ea552cb334fea3122e372198508e3b6bbbc874eefa3 boot boot.obj
c40a8338903a1acae3013ec38326f540e901a6d55d63a32fafee16e96053f27e boat boat.obj
17200138dad1ad0f86ef9a98dd38f9932098b2c9e94d8694bd3f6143601e8c54 armadillo armadillo.obj
16173477cbbf93c8126f5150db642f561b06d736d047d429ea45a97d145dbf4d penguin penguin_hr.obj
5fabbceb4fb27534f0a15077e305b65bf0fcab83f188c90fd71a9c86f9ca7e3a spot spot.obj
9fb9b91389918ac8fddd11a44ca2d35350e200f92a2151b4e892841e49aaa158 fish fish.obj
cdf398872bd22c86a12ac65265aef1ea939af4332f8d259219fe7ea5a1cf272c plane plane.obj
cb2fb353cca21db7b9dcdef96e3fc01dc42489c24b69c64c1ea0a0741d16edc4 hand hand_closed.obj
bad29482ce08f737912f544e3b1bef030a180e5e2efa211753a0b6a6606cab41 human human_man.obj
73bf1247cbe3036057fa7b057e213764bb609af1e9ba8e7005126353c2845595 human human_woman.obj
ff8d168e9cf43194d3821eef1f6efeac5149eed8b1578e6d97bc6a6aba95254a wingnut wingnut.obj
b0aa414ce3c46e5e30ee506261f1e3cb85025e2a33b5b18bc638b08a7c57a6cb goathead goathead.obj
f7ae0f03f26c952d208658ba70923687f67ddc7351adf15f91da7aa61dbcd119 torus torus.obj
LIST
exit $STATUS
