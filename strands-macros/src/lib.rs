use proc_macro::TokenStream;

/// Attribute macro for defining agent tools from async functions.
///
/// # Example
///
/// ```ignore
/// #[tool]
/// async fn get_weather(
///     /// The city to check weather for
///     city: String,
/// ) -> Result<String, strands_core::StrandsError> {
///     Ok(format!("22 degrees in {city}"))
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    tool_impl(item.into()).into()
}

fn tool_impl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let func: syn::ItemFn = match syn::parse2(input.clone()) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error(),
    };

    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Build PascalCase struct name
    let struct_name_str = to_pascal_case(&fn_name_str);
    let struct_name = syn::Ident::new(&struct_name_str, fn_name.span());

    // Extract doc comment from function for tool description
    let description = extract_doc_comment(&func.attrs);

    // Parse parameters (skip self if present)
    let params: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg {
                let name = match pat_type.pat.as_ref() {
                    syn::Pat::Ident(ident) => ident.ident.to_string(),
                    _ => return None,
                };
                let doc = extract_doc_comment(&pat_type.attrs);
                let ty = &pat_type.ty;
                let is_option = is_option_type(ty);
                let json_type = rust_type_to_json_type(ty);
                Some(ParamInfo {
                    name,
                    doc,
                    is_option,
                    json_type,
                    ty: ty.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    // Build JSON schema properties
    let schema_properties: Vec<proc_macro2::TokenStream> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            let json_type = &p.json_type;
            let desc = &p.doc;
            quote::quote! {
                properties.insert(
                    #name.to_string(),
                    serde_json::json!({
                        "type": #json_type,
                        "description": #desc
                    }),
                );
            }
        })
        .collect();

    let required_params: Vec<proc_macro2::TokenStream> = params
        .iter()
        .filter(|p| !p.is_option)
        .map(|p| {
            let name = &p.name;
            quote::quote! { #name.to_string() }
        })
        .collect();

    // Build parameter extraction in invoke()
    let param_extractions: Vec<proc_macro2::TokenStream> = params
        .iter()
        .map(|p| {
            let name_str = &p.name;
            let name_ident = syn::Ident::new(&p.name, proc_macro2::Span::call_site());
            let ty = &p.ty;
            if p.is_option {
                quote::quote! {
                    let #name_ident: #ty = input.get(#name_str)
                        .and_then(|v| serde_json::from_value(v.clone()).ok());
                }
            } else {
                quote::quote! {
                    let #name_ident: #ty = serde_json::from_value(
                        input.get(#name_str)
                            .cloned()
                            .ok_or_else(|| strands_core::StrandsError::Tool {
                                tool_name: #fn_name_str.to_string(),
                                message: format!("Missing required parameter: {}", #name_str),
                            })?
                    ).map_err(|e| strands_core::StrandsError::Tool {
                        tool_name: #fn_name_str.to_string(),
                        message: format!("Invalid parameter {}: {}", #name_str, e),
                    })?;
                }
            }
        })
        .collect();

    let param_names: Vec<syn::Ident> = params
        .iter()
        .map(|p| syn::Ident::new(&p.name, proc_macro2::Span::call_site()))
        .collect();

    let output = quote::quote! {
        // Keep the original function
        #func

        pub struct #struct_name;

        #[async_trait::async_trait]
        impl strands_core::Tool for #struct_name {
            fn name(&self) -> &str {
                #fn_name_str
            }

            fn spec(&self) -> strands_core::types::tools::ToolSpec {
                let mut properties = serde_json::Map::new();
                #(#schema_properties)*

                strands_core::types::tools::ToolSpec {
                    name: #fn_name_str.to_string(),
                    description: #description.to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": serde_json::Value::Object(properties),
                        "required": vec![#(#required_params),*]
                    }),
                }
            }

            async fn invoke(
                &self,
                input: serde_json::Value,
                _ctx: &strands_core::ToolContext,
            ) -> strands_core::Result<strands_core::ToolOutput> {
                #(#param_extractions)*

                let result = #fn_name(#(#param_names),*).await?;
                let content = serde_json::to_value(result)
                    .map_err(|e| strands_core::StrandsError::Tool {
                        tool_name: #fn_name_str.to_string(),
                        message: e.to_string(),
                    })?;
                Ok(strands_core::ToolOutput {
                    content,
                    is_error: false,
                })
            }
        }
    };

    output
}

struct ParamInfo {
    name: String,
    doc: String,
    is_option: bool,
    json_type: String,
    ty: Box<syn::Type>,
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().to_string() + &chars.collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

fn extract_doc_comment(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let syn::Lit::Str(s) = &expr_lit.lit {
                            return Some(s.value().trim().to_string());
                        }
                    }
                }
            }
            None
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

fn rust_type_to_json_type(ty: &syn::Type) -> String {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let ident = segment.ident.to_string();
            return match ident.as_str() {
                "String" | "str" => "string",
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "isize"
                | "usize" => "integer",
                "f32" | "f64" => "number",
                "bool" => "boolean",
                "Vec" => "array",
                "Option" => {
                    // Unwrap the inner type
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            return rust_type_to_json_type(inner);
                        }
                    }
                    "string"
                }
                _ => "object",
            }
            .to_string();
        }
    }
    "string".to_string()
}
