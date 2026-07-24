# Third-party notices

PhoneLM original source code and documentation are licensed under the Apache License 2.0 unless otherwise noted.

## Qualcomm QAIRT / QNN

Qualcomm QAIRT and QNN SDK headers, runtime libraries, Stub libraries, Skel libraries, and related tools are external prerequisites governed by their own license terms. They are not tracked in this repository and are not relicensed under PhoneLM's Apache License 2.0. APKs built with locally supplied Qualcomm components are build artifacts and are not distributed by this repository.

## MNN

The build can fetch MNN 3.5.0 at commit `c35f14f3ab5cb65094863b9a0e888370b027a670`. Its source tree is intentionally ignored and is governed by MNN's original Apache License 2.0 and upstream notices. PhoneLM does not relicense MNN.

## Gradle wrapper

The tracked Gradle wrapper scripts and `gradle/wrapper/gradle-wrapper.jar` are third-party Gradle components distributed under the Apache License 2.0. Their existing copyright and license notices remain in effect.

## Other build dependencies

Android, Kotlin, Android Gradle Plugin, and other dependencies downloaded by the build are governed by their respective upstream licenses. They are not relicensed by PhoneLM.

No Qualcomm binary, QAIRT/QNN header, APK, AAB, signing key, MNN source tree, or generated native library is covered by PhoneLM's license merely because it is used during a local build.
