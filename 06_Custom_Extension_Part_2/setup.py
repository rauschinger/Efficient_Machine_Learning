import setuptools
import torch.utils.cpp_extension

setuptools.setup(   name = "eml_ext",
                    ext_modules = [ torch.utils.cpp_extension.CppExtension( "eml_ext_hello_world_cpp", 
                                                                            ["eml/ext/HelloWorld.cpp"]
                                                                          ),
                                    torch.utils.cpp_extension.CppExtension( "eml_ext_linear_cpp", 
                                                                            ["eml/ext/LinearCpp.cpp"]
                                                                          )
                                  ],
                    cmdclass = { 'build_ext': torch.utils.cpp_extension.BuildExtension}
                )
