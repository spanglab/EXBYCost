"""
Biopolymer Repeating Unit SMILES Lookup Tool

Looks up SMILES of repeating units in biopolymers using the PubChem API
via the pubchempy library.

No built-in database - all lookups are done dynamically via API searches.
Uses ChEBI (Chemical Entities of Biological Interest) to find chemical names 
and relationships for polymers.

Install: pip install pubchempy requests
"""

import pubchempy as pcp
import requests
from typing import Optional, List, Dict, Any


class BiopolymerSMILESLookup:
    """Class to look up SMILES of biopolymer repeating units via PubChem and ChEBI."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BiopolymerSMILESLookup/1.0 (Educational/Research Use)',
            'Accept': 'application/json'
        })

    def _log(self, message: str):
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [DEBUG] {message}")

    def _get_compounds_by_name(self, name: str) -> List[pcp.Compound]:
        """Search PubChem for compounds by name."""
        try:
            compounds = pcp.get_compounds(name, 'name')
            self._log(f"PubChem '{name}': found {len(compounds)} compounds")
            return compounds
        except Exception as e:
            self._log(f"PubChem '{name}': ERROR - {e}")
            return []

    def _search_chebi(self, term: str) -> List[Dict[str, Any]]:
        """
        Search ChEBI via the EBI OLS API.
        Returns list of matching ChEBI entities.
        """
        url = "https://www.ebi.ac.uk/ols4/api/search"
        params = {
            "q": term,
            "ontology": "chebi",
            "rows": 10,
            "exact": "false"
        }

        self._log(f"ChEBI search: '{term}'")
        self._log(f"  URL: {url}")
        self._log(f"  Params: {params}")

        try:
            response = self.session.get(url, params=params, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                self._log(f"  Found {len(docs)} results")
                for i, doc in enumerate(docs[:3]):
                    self._log(f"    [{i}] {doc.get('label', 'N/A')} ({doc.get('obo_id', 'N/A')})")
                return docs
            else:
                self._log(f"  Response body: {response.text[:500]}")
        except Exception as e:
            self._log(f"  ERROR: {e}")
        return []

    def _get_chebi_entity(self, chebi_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed ChEBI entity information including relationships.
        """
        # Clean up the ChEBI ID
        if chebi_id.startswith("http"):
            chebi_id = chebi_id.split("/")[-1]
        if not chebi_id.startswith("CHEBI:"):
            chebi_id = f"CHEBI:{chebi_id}"

        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms"
        params = {
            "iri": f"http://purl.obolibrary.org/obo/{chebi_id.replace(':', '_')}"
        }

        self._log(f"ChEBI entity lookup: {chebi_id}")

        try:
            response = self.session.get(url, params=params, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                terms = data.get("_embedded", {}).get("terms", [])
                self._log(f"  Found {len(terms)} terms")
                if terms:
                    return terms[0]
            else:
                self._log(f"  Response body: {response.text[:500]}")
        except Exception as e:
            self._log(f"  ERROR: {e}")
        return None

    def _get_chebi_relations(self, chebi_iri: str) -> Dict[str, List[str]]:
        """
        Get relationships for a ChEBI entity (has_part, has_functional_parent, etc.)
        Returns dict mapping relation type to list of related entity labels.
        """
        relations = {
            "has_part": [],
            "has_functional_parent": [],
            "has_parent_hydride": [],
            "is_conjugate_acid_of": [],
            "is_conjugate_base_of": [],
        }

        # Get the term details which includes relationships
        encoded_iri = requests.utils.quote(requests.utils.quote(chebi_iri, safe=''), safe='')
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/{encoded_iri}"

        self._log(f"ChEBI relations lookup: {chebi_iri}")
        self._log(f"  URL: {url}")

        try:
            response = self.session.get(url, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # Check for relation links
                links = data.get("_links", {})
                self._log(f"  Available links: {list(links.keys())}")

                # Method 1: Check if there are direct relation links (like has_functional_parent)
                for rel_type in relations.keys():
                    rel_key = rel_type.replace("_", " ")  # "has_part" -> "has part"
                    if rel_type in links or rel_key in links:
                        link_key = rel_type if rel_type in links else rel_key
                        rel_url = links[link_key].get("href", "")
                        if rel_url:
                            self._log(f"  Found direct relation link: {link_key}")
                            try:
                                rel_response = self.session.get(rel_url, timeout=15)
                                if rel_response.status_code == 200:
                                    rel_data = rel_response.json()
                                    # Handle both embedded terms and direct response
                                    terms = rel_data.get("_embedded", {}).get("terms", [])
                                    if not terms and isinstance(rel_data, list):
                                        terms = rel_data
                                    for term in terms:
                                        label = term.get("label", "")
                                        if label:
                                            relations[rel_type].append(label)
                                            self._log(f"    Found: {label}")
                            except Exception as e:
                                self._log(f"    Error fetching relation: {e}")

                # Method 2: Try to get graph relationships
                if "graph" in links:
                    graph_url = links["graph"]["href"]
                    self._log(f"  Fetching graph: {graph_url}")
                    graph_response = self.session.get(graph_url, timeout=15)
                    self._log(f"  Graph response status: {graph_response.status_code}")

                    if graph_response.status_code == 200:
                        graph_data = graph_response.json()
                        edges = graph_data.get("edges", [])
                        nodes_list = graph_data.get("nodes", [])

                        self._log(f"  Graph has {len(nodes_list)} nodes, {len(edges)} edges")

                        # Build nodes dict - handle different possible structures
                        nodes = {}
                        for n in nodes_list:
                            # Try different possible id fields
                            node_id = n.get("id") or n.get("iri") or n.get("uri") or n.get("obo_id")
                            node_label = n.get("label") or n.get("name") or ""
                            if node_id:
                                nodes[node_id] = node_label

                        self._log(f"  Built nodes dict with {len(nodes)} entries")

                        for edge in edges:
                            # Get edge properties - handle different structures
                            rel_type = edge.get("label", "") or edge.get("property", "") or edge.get("predicate", "")
                            rel_type = rel_type.replace(" ", "_")

                            target = edge.get("target") or edge.get("object") or edge.get("to")
                            source = edge.get("source") or edge.get("subject") or edge.get("from")

                            target_label = nodes.get(target, "") or target

                            self._log(f"    Edge: {rel_type}: {source} -> {target_label}")

                            if rel_type in relations and target_label:
                                relations[rel_type].append(target_label)
                else:
                    self._log(f"  No 'graph' link available")
            else:
                self._log(f"  Response body: {response.text[:500]}")
        except Exception as e:
            self._log(f"  ERROR: {e}")
            import traceback
            self._log(f"  Traceback: {traceback.format_exc()}")

        self._log(f"  Relations found: {relations}")
        return relations

    def _get_chebi_smiles(self, chebi_id: str) -> Optional[str]:
        """
        Get SMILES directly from ChEBI using multiple methods.
        """
        if not chebi_id.startswith("CHEBI:"):
            chebi_id = f"CHEBI:{chebi_id}"

        # Method 1: Try the main ChEBI web service (not test)
        url = "https://www.ebi.ac.uk/webservices/chebi/2.0/getCompleteEntity"
        params = {"chebiId": chebi_id}

        self._log(f"ChEBI SMILES lookup (method 1): {chebi_id}")
        self._log(f"  URL: {url}")

        try:
            response = self.session.get(url, params=params, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                import re
                smiles_match = re.search(r'<smiles>([^<]+)</smiles>', response.text)
                if smiles_match:
                    smiles = smiles_match.group(1)
                    self._log(f"  Found SMILES: {smiles[:50]}...")
                    return smiles
                else:
                    self._log(f"  No SMILES tag found in response")
            else:
                self._log(f"  Method 1 failed: {response.status_code}")
        except Exception as e:
            self._log(f"  Method 1 ERROR: {e}")

        # Method 2: Try OLS4 term endpoint which may have SMILES in annotations
        chebi_num = chebi_id.replace("CHEBI:", "")
        iri = f"http://purl.obolibrary.org/obo/CHEBI_{chebi_num}"
        encoded_iri = requests.utils.quote(requests.utils.quote(iri, safe=''), safe='')
        url2 = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/{encoded_iri}"

        self._log(f"ChEBI SMILES lookup (method 2 - OLS4): {chebi_id}")
        self._log(f"  URL: {url2}")

        try:
            response = self.session.get(url2, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                # Check annotation field for SMILES
                annotation = data.get("annotation", {})
                self._log(f"  Annotation keys: {list(annotation.keys()) if annotation else 'None'}")

                # Try different possible keys for SMILES
                for key in ["smiles", "SMILES", "smiles_string", "has_smiles", "chebi_smiles"]:
                    if key in annotation:
                        smiles_list = annotation[key]
                        if smiles_list and len(smiles_list) > 0:
                            smiles = smiles_list[0]
                            self._log(f"  Found SMILES in annotation[{key}]: {smiles[:50]}...")
                            return smiles

                self._log(f"  No SMILES found in annotations")
        except Exception as e:
            self._log(f"  Method 2 ERROR: {e}")

        # Method 3: Try the libchebipy-style direct API
        url3 = f"https://www.ebi.ac.uk/chebi/saveStructure.do?chebiId={chebi_num}&imageId=0&smilesType=all&outputType=smiles"

        self._log(f"ChEBI SMILES lookup (method 3 - saveStructure): {chebi_id}")
        self._log(f"  URL: {url3}")

        try:
            response = self.session.get(url3, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                smiles = response.text.strip()
                if smiles and len(smiles) > 0 and not smiles.startswith("<!") and not smiles.startswith("<"):
                    self._log(f"  Found SMILES: {smiles[:50]}...")
                    return smiles
                else:
                    self._log(f"  Response not a valid SMILES: {smiles[:100]}")
        except Exception as e:
            self._log(f"  Method 3 ERROR: {e}")

        return None

    def _get_chebi_children(self, chebi_iri: str) -> List[Dict[str, Any]]:
        """
        Get children (subtypes) of a ChEBI entity.
        Returns list of child entities with their labels and IDs.
        """
        children = []

        encoded_iri = requests.utils.quote(requests.utils.quote(chebi_iri, safe=''), safe='')
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/{encoded_iri}/children"

        self._log(f"ChEBI children lookup: {chebi_iri}")
        self._log(f"  URL: {url}")

        try:
            response = self.session.get(url, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                self._log(f"  Response keys: {list(data.keys())}")

                # Debug: print raw response structure
                if "_embedded" in data:
                    self._log(f"  _embedded keys: {list(data['_embedded'].keys())}")

                terms = data.get("_embedded", {}).get("terms", [])
                self._log(f"  Found {len(terms)} children")

                for term in terms:
                    label = term.get("label", "")
                    obo_id = term.get("obo_id", "")
                    iri = term.get("iri", "")
                    self._log(f"    Child: {label} ({obo_id})")

                    if label and obo_id:
                        children.append({
                            "label": label,
                            "obo_id": obo_id,
                            "iri": iri
                        })
            else:
                self._log(f"  Response body: {response.text[:500]}")
        except Exception as e:
            self._log(f"  ERROR: {e}")
            import traceback
            self._log(f"  Traceback: {traceback.format_exc()}")

        self._log(f"  Returning {len(children)} children")
        return children

    def _get_chebi_parents(self, chebi_iri: str) -> List[Dict[str, Any]]:
        """
        Get parents (supertypes) of a ChEBI entity.
        Returns list of parent entities with their labels and IDs.
        """
        parents = []

        encoded_iri = requests.utils.quote(requests.utils.quote(chebi_iri, safe=''), safe='')
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/{encoded_iri}/parents"

        self._log(f"ChEBI parents lookup: {chebi_iri}")
        self._log(f"  URL: {url}")

        try:
            response = self.session.get(url, timeout=15)
            self._log(f"  Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                terms = data.get("_embedded", {}).get("terms", [])
                self._log(f"  Found {len(terms)} parents")

                for term in terms:
                    label = term.get("label", "")
                    obo_id = term.get("obo_id", "")
                    iri = term.get("iri", "")
                    self._log(f"    Parent: {label} ({obo_id})")

                    if label and obo_id:
                        parents.append({
                            "label": label,
                            "obo_id": obo_id,
                            "iri": iri
                        })
        except Exception as e:
            self._log(f"  ERROR: {e}")

        self._log(f"  Returning {len(parents)} parents")
        return parents

    def _find_chebi_alternatives(self, term: str) -> List[Dict[str, Any]]:
        """
        Search ChEBI for a term and find related monomers/parts.
        Returns list of alternative compounds to search.
        """
        alternatives = []

        self._log(f"Finding ChEBI alternatives for: '{term}'")

        # Search ChEBI for the term
        search_results = self._search_chebi(term)

        if not search_results:
            self._log(f"No ChEBI search results for '{term}'")

        for result in search_results[:3]:  # Check top 3 results
            chebi_id = result.get("obo_id", "")
            label = result.get("label", "")
            iri = result.get("iri", "")

            self._log(f"Processing ChEBI result: {label} ({chebi_id})")

            if not chebi_id:
                self._log(f"  Skipping - no obo_id")
                continue

            # Try to get SMILES directly from ChEBI
            smiles = self._get_chebi_smiles(chebi_id)
            if smiles:
                self._log(f"  Got SMILES directly from ChEBI")
                alternatives.append({
                    "name": label,
                    "chebi_id": chebi_id,
                    "smiles": smiles,
                    "source": "chebi_direct"
                })
            else:
                self._log(f"  No SMILES from ChEBI, adding label for PubChem search")
                # Still add the label so we can try PubChem
                alternatives.append({
                    "name": label,
                    "chebi_id": chebi_id,
                    "source": "chebi_label"
                })

            # Get the entity IRI for relationship lookup
            if iri:
                self._log(f"  Looking up relations for IRI: {iri}")
                relations = self._get_chebi_relations(iri)

                # Add related compounds (these are likely monomers)
                # Filter out self-references (where the relation points back to the same entity)
                found_parts = False
                for rel_type in ["has_part", "has_functional_parent", "has_parent_hydride"]:
                    for related_name in relations.get(rel_type, []):
                        # Skip if it's a self-reference
                        if related_name.lower() == label.lower():
                            self._log(f"  Skipping self-reference: {rel_type} -> {related_name}")
                            continue
                        found_parts = True
                        self._log(f"  Found relation: {rel_type} -> {related_name}")
                        alternatives.append({
                            "name": related_name,
                            "relation": rel_type,
                            "parent": label,
                            "source": "chebi_relation"
                        })

                self._log(f"  found_parts = {found_parts}")

                # If no parts found, look at children (subtypes that might have SMILES)
                if not found_parts:
                    self._log(f"  No parts found, checking children...")
                    children = self._get_chebi_children(iri)
                    self._log(f"  Got {len(children)} children back")

                    for child in children[:5]:  # Limit to first 5 children
                        child_label = child.get("label", "")
                        child_id = child.get("obo_id", "")
                        self._log(f"  Processing child: {child_label} ({child_id})")

                        # Try to get SMILES for this child
                        child_smiles = self._get_chebi_smiles(child_id)
                        if child_smiles:
                            self._log(f"  Found child with SMILES: {child_label}")
                            alternatives.append({
                                "name": child_label,
                                "chebi_id": child_id,
                                "smiles": child_smiles,
                                "relation": "is_a",
                                "parent": label,
                                "source": "chebi_child"
                            })
                        else:
                            # Still add for PubChem search
                            self._log(f"  Adding child for PubChem search: {child_label}")
                            alternatives.append({
                                "name": child_label,
                                "chebi_id": child_id,
                                "relation": "is_a",
                                "parent": label,
                                "source": "chebi_child_label"
                            })

                    # Also check parents (the entity IS A parent, parent might have SMILES)
                    if not alternatives or not any(a.get("smiles") for a in alternatives):
                        self._log(f"  No SMILES from children, checking parents...")
                        parents = self._get_chebi_parents(iri)
                        self._log(f"  Got {len(parents)} parents back")

                        for parent_entity in parents[:5]:  # Limit to first 5 parents
                            parent_label = parent_entity.get("label", "")
                            parent_id = parent_entity.get("obo_id", "")
                            self._log(f"  Processing parent: {parent_label} ({parent_id})")

                            # Try to get SMILES for this parent
                            parent_smiles = self._get_chebi_smiles(parent_id)
                            if parent_smiles:
                                self._log(f"  Found parent with SMILES: {parent_label}")
                                alternatives.append({
                                    "name": parent_label,
                                    "chebi_id": parent_id,
                                    "smiles": parent_smiles,
                                    "relation": "subclass_of",
                                    "child": label,
                                    "source": "chebi_parent"
                                })
                            else:
                                # Still add for PubChem search
                                self._log(f"  Adding parent for PubChem search: {parent_label}")
                                alternatives.append({
                                    "name": parent_label,
                                    "chebi_id": parent_id,
                                    "relation": "subclass_of",
                                    "child": label,
                                    "source": "chebi_parent_label"
                                })
            else:
                self._log(f"  No IRI available for relations lookup")

        self._log(f"Total alternatives found: {len(alternatives)}")
        return alternatives

    def _format_compound(self, search_term: str, compound: pcp.Compound) -> Dict[str, Any]:
        """Format a PubChemPy Compound into a result dictionary."""
        return {
            "search_term": search_term,
            "cid": compound.cid,
            "smiles": compound.smiles,
            "connectivity_smiles": compound.connectivity_smiles,
            "iupac_name": compound.iupac_name,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}",
            "source": "pubchem"
        }

    def lookup(self, polymer_name: str) -> Dict[str, Any]:
        """
        Look up the repeating unit (monomer) SMILES for a biopolymer.

        Tries multiple search strategies:
        1. Direct search on PubChem
        2. Direct ChEBI search for SMILES
        3. Search ChEBI for relationships and children
        4. Search PubChem with names found from ChEBI

        Args:
            polymer_name: Name of the biopolymer (e.g., "cellulose", "chitin")

        Returns:
            Dictionary with polymer name and list of found monomers with SMILES
        """
        self._log(f"=== Starting lookup for: {polymer_name} ===")

        results = {
            "polymer": polymer_name,
            "monomers": [],
            "search_strategies_used": [],
            "chebi_alternatives_found": []
        }

        seen_cids = set()
        seen_smiles = set()

        # Strategy 1: Direct PubChem searches
        self._log("Strategy 1: Direct PubChem searches")
        search_terms = [
            polymer_name,
            f"{polymer_name} monomer",
            f"{polymer_name} repeating unit",
        ]

        for search_term in search_terms:
            compounds = self._get_compounds_by_name(search_term)

            if compounds:
                strategy_name = search_term.replace(polymer_name, "").strip() or "direct"
                results["search_strategies_used"].append(f"pubchem_{strategy_name}")

                for compound in compounds[:3]:
                    if compound.cid and compound.cid not in seen_cids:
                        seen_cids.add(compound.cid)
                        results["monomers"].append(
                            self._format_compound(search_term, compound)
                        )

        self._log(f"After PubChem: found {len(results['monomers'])} monomers")

        # If PubChem found results, we're done
        if results["monomers"]:
            self._log("PubChem found results, skipping ChEBI lookup")
            return results

        # Strategy 2: Direct ChEBI SMILES lookup
        self._log("Strategy 2: Direct ChEBI SMILES lookup")
        chebi_search_results = self._search_chebi(polymer_name)

        for result in chebi_search_results[:3]:
            chebi_id = result.get("obo_id", "")
            label = result.get("label", "")

            if chebi_id:
                smiles = self._get_chebi_smiles(chebi_id)
                if smiles and smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    self._log(f"Found SMILES directly from ChEBI: {label}")
                    results["search_strategies_used"].append("chebi_direct")
                    results["monomers"].append({
                        "search_term": label,
                        "smiles": smiles,
                        "chebi_id": chebi_id,
                        "source": "chebi_direct",
                        "chebi_url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_id}"
                    })

        # If we found SMILES directly from ChEBI, we're done
        if results["monomers"]:
            self._log("ChEBI direct lookup found SMILES, skipping relationship lookup")
            return results

        # Strategy 3: Search ChEBI for relationships and children
        self._log("Strategy 3: ChEBI relationships and children lookup")
        chebi_alternatives = self._find_chebi_alternatives(polymer_name)
        results["chebi_alternatives_found"] = [
            {"name": a.get("name"), "relation": a.get("relation"), "source": a.get("source")}
            for a in chebi_alternatives
        ]

        if chebi_alternatives:
            results["search_strategies_used"].append("chebi")
            self._log(f"Found {len(chebi_alternatives)} ChEBI alternatives")

            for alt in chebi_alternatives:
                # If ChEBI provided SMILES directly, add it
                if alt.get("smiles") and alt["smiles"] not in seen_smiles:
                    seen_smiles.add(alt["smiles"])
                    self._log(f"Adding ChEBI SMILES for: {alt['name']}")
                    results["monomers"].append({
                        "search_term": alt["name"],
                        "smiles": alt["smiles"],
                        "chebi_id": alt.get("chebi_id"),
                        "source": alt.get("source", "chebi"),
                        "relation": alt.get("relation"),
                        "chebi_url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={alt.get('chebi_id', '')}"
                    })

                # Also try searching PubChem with the ChEBI name
                alt_name = alt.get("name", "")
                if alt_name:
                    self._log(f"Searching PubChem for ChEBI name: {alt_name}")
                    compounds = self._get_compounds_by_name(alt_name)
                    for compound in compounds[:2]:
                        if compound.cid and compound.cid not in seen_cids:
                            seen_cids.add(compound.cid)
                            result = self._format_compound(alt_name, compound)
                            result["chebi_relation"] = alt.get("relation")
                            results["monomers"].append(result)
        else:
            self._log("No ChEBI alternatives found")

        self._log(f"=== Lookup complete: {len(results['monomers'])} total monomers ===")
        return results

    def get_smiles(self, polymer_name: str) -> Optional[str]:
        """
        Simple function to get just the SMILES string.

        Args:
            polymer_name: Name of the biopolymer

        Returns:
            SMILES string of the primary monomer, or None if not found
        """
        result = self.lookup(polymer_name)
        if result["monomers"]:
            return result["monomers"][0]["smiles"]
        return None


def get_biopolymer_smiles(polymer_name: str, verbose: bool = False) -> Optional[str]:
    """
    Get the SMILES of a biopolymer's repeating unit.

    Args:
        polymer_name: Name of the biopolymer (e.g., "cellulose", "chitin")
        verbose: If True, print debug information

    Returns:
        SMILES string or None if not found
    """
    lookup = BiopolymerSMILESLookup(verbose=verbose)
    return lookup.get_smiles(polymer_name)


def print_results(results: Dict[str, Any]) -> None:
    """Pretty print the lookup results."""
    print(f"\n{'='*60}")
    print(f"Biopolymer: {results['polymer']}")
    if results['search_strategies_used']:
        print(f"Search strategies: {', '.join(results['search_strategies_used'])}")
    if results.get('chebi_alternatives_found'):
        print(f"ChEBI alternatives: {[a['name'] for a in results['chebi_alternatives_found'][:5]]}")
    print(f"{'='*60}")

    if not results["monomers"]:
        print("No monomers found. Try a different search term.")
        return

    for i, monomer in enumerate(results["monomers"], 1):
        print(f"\nResult {i}: {monomer['search_term']}")
        print(f"  SMILES: {monomer.get('smiles', 'N/A')}")
        if monomer.get('connectivity_smiles'):
            print(f"  Connectivity SMILES: {monomer.get('connectivity_smiles', 'N/A')}")
        if monomer.get('iupac_name'):
            print(f"  IUPAC Name: {monomer.get('iupac_name', 'N/A')}")
        if monomer.get('molecular_formula'):
            print(f"  Formula: {monomer.get('molecular_formula', 'N/A')}")
        if monomer.get('molecular_weight'):
            print(f"  MW: {monomer.get('molecular_weight', 'N/A')} g/mol")
        if monomer.get('cid'):
            print(f"  PubChem CID: {monomer['cid']}")
            print(f"  URL: {monomer.get('pubchem_url', 'N/A')}")
        if monomer.get('chebi_id'):
            print(f"  ChEBI ID: {monomer['chebi_id']}")
            print(f"  URL: {monomer.get('chebi_url', 'N/A')}")
        if monomer.get('relation'):
            print(f"  Relation: {monomer['relation']}")
        print(f"  Source: {monomer.get('source', 'unknown')}")


if __name__ == "__main__":
    # Test with various biopolymers
    test_compounds = [
        # Direct monomers (should work directly on PubChem)
        "glucose",
        "xylose",
        "N-acetylglucosamine",
        # Polymers (should use ChEBI lookup)
        "cellulose",
        "chitin",
        "xylan",
        "starch",
        "pectin",
        "alginate",
        "hyaluronic acid",
        "dextran",
        "pullulan",
        "mannan",
        "arabinan",
        "galactan",
        "inulin",
        "curdlan",
        "laminarin",
        "lichenan",
        "glycogen",
        "amylose",
        "amylopectin",
        "chitosan",
        "heparin",
        "chondroitin sulfate",
        "keratan sulfate",
        "agarose",
        "carrageenan",
        "fucoidan",
        'cyanidin-3-pentoside'
    ]

    print("Biopolymer Repeating Unit SMILES Lookup")
    print("Using PubChem + ChEBI for dynamic name resolution")
    print("=" * 60)

    lookup = BiopolymerSMILESLookup(verbose=False)

    for compound in test_compounds:
        result = lookup.lookup(compound)

        if result["monomers"]:
            monomer = result["monomers"][0]
            smiles = monomer["smiles"]
            search_term = monomer["search_term"]
            source = monomer.get("source", "unknown")
            print(f"{compound}: {smiles}")
            print(f"  (found via: {search_term}, source: {source})")
        else:
            print(f"{compound}: NOT FOUND")